"""SoftVLA model wrapper for RoboTwin evaluation.

Mirrors the PI0 class in policy/pi05/pi_model.py but loads SoftVLAPytorch
(PyTorch-based) instead of the JAX PI0 model.

Checkpoint layout expected under policy/Soft-VLA/checkpoints/:
    {train_config_name}/{model_name}/{checkpoint_id}/
        model.safetensors
        assets/
            {asset_id}/
                norm_stats files
"""

import logging
import os
import sys

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn

# ── resolve paths ──────────────────────────────────────────────────────────────
# This file lives at  <repo>/third_party/RoboTwin/policy/Soft-VLA/softvla_model.py.
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_POLICY_DIR, "..", "..", "..", ".."))
_SRC_DIR    = os.path.join(_REPO_ROOT, "src")
_OPENPI_CLIENT_DIR = os.path.join(_REPO_ROOT, "packages", "openpi-client", "src")

if not os.path.isdir(os.path.join(_SRC_DIR, "softvla")):
    raise ModuleNotFoundError(
        f"Soft-VLA src not found at {_SRC_DIR}. Expected layout: <repo>/src/softvla/."
    )

for _p in (_SRC_DIR, _OPENPI_CLIENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from softvla.models.softvla_config import SoftVLAConfig          # noqa: E402
from softvla.models.softvla_pytorch import SoftVLAPytorch        # noqa: E402
from openpi.policies import policy as _policy                    # noqa: E402
from openpi.training import config as _config                    # noqa: E402
from openpi.shared import normalize as _normalize                # noqa: E402
import openpi.transforms as _transforms                          # noqa: E402

logger = logging.getLogger(__name__)


# ── image float-cast transform ─────────────────────────────────────────────────
# Bugfix: Observation.from_dict applies permute(0,3,1,2) for torch.uint8 images
# (BHWC→BCHW), but preprocessing_pytorch.py then converts back to BHWC and
# fails to convert BACK to BCHW at the end (the `if not is_channels_first` guard
# is evaluated on the *original* shape, not the post-permute shape).
# Avoiding torch.uint8 in the Observation entirely sidesteps this mismatch:
# float32 images are left untouched by from_dict and preprocessing_pytorch
# correctly converts them BHWC→BCHW at the end.

import dataclasses as _dataclasses  # noqa: E402 (already stdlib, import here for clarity)

@_dataclasses.dataclass(frozen=True)
class _CastImagesToFloat(_transforms.DataTransformFn):
    """Convert uint8 images in data["image"] to float32 in [-1, 1] range.

    Inserted as the last input transform so that images arrive at
    Observation.from_dict as float32 tensors, bypassing the torch.uint8 path
    that incorrectly applies permute(0, 3, 1, 2).
    """

    def __call__(self, data: dict) -> dict:
        if "image" not in data:
            return data
        new_images = {}
        for k, v in data["image"].items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.unsignedinteger):
                arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
            new_images[k] = arr
        data = dict(data)
        data["image"] = new_images
        return data


# ── domain_ids wrapper ─────────────────────────────────────────────────────────

class _SoftVLAInferenceWrapper(nn.Module):
    """Thin nn.Module wrapper that injects domain_ids into SoftVLAPytorch.sample_actions.

    Policy.infer() calls  model.sample_actions(device, observation, **sample_kwargs).
    SoftVLAPytorch.sample_actions requires an additional domain_ids tensor and
    num_steps int that are not present in the standard Policy.infer() interface.
    This wrapper captures them so the standard interface works unchanged.
    """

    def __init__(self, model: SoftVLAPytorch, domain_id: int, num_steps: int):
        super().__init__()
        self.wrapped = model
        self.domain_id = domain_id
        self.num_steps = num_steps

    def sample_actions(self, device: str, observation, **kwargs) -> torch.Tensor:
        domain_ids = torch.tensor([self.domain_id], dtype=torch.long, device=device)
        # The model is loaded as bfloat16 but inputs (noise, state, timestep) arrive
        # as float32.  autocast promotes matmul inputs to the model's native dtype
        # automatically, avoiding the "Float vs BFloat16" runtime error.
        device_type = device.split(":")[0]  # e.g. "cuda" from "cuda:0"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return self.wrapped.sample_actions(
                device,
                observation,
                domain_ids=domain_ids,
                num_steps=self.num_steps,
            )


# ── SoftVLA class ──────────────────────────────────────────────────────────────

class SoftVLA:
    """SoftVLA policy wrapper for RoboTwin evaluation.

    Args:
        train_config_name: Name of the TrainConfig registered in openpi/training/config.py
                           (e.g. "softvla_robotwin").  Used to obtain data transforms.
                           If a fine-tune checkpoint root uses a suffix (e.g.
                           "softvla_robotwin_ft"), this class will automatically
                           fall back to the base TrainConfig name when resolving
                           OpenPI transforms.
        model_name:        Experiment / checkpoint name (directory under train_config_name).
        checkpoint_id:     Checkpoint step number (sub-directory under model_name).
        softvla_step:      Number of actions to execute per inference call (action chunk
                           execution length).  Matches the semantics of pi0_step in PI0.
                           E.g. softvla_step=8 executes only the first 8 actions from the
                           predicted action_horizon-length chunk.
        num_denoise_steps: Number of flow-matching ODE integration steps used when generating
                           actions.  Higher values give higher-quality actions but increase
                           inference latency.  Default 10.
        domain_id:         Index into SoftPromptHub; selects the embodiment-specific soft prompt.
                           Default 0 works when only one embodiment is registered.
    """

    def __init__(
        self,
        train_config_name: str,
        model_name: str,
        checkpoint_id: int,
        softvla_step: int,
        num_denoise_steps: int = 10,
        domain_id: int = 0,
    ):
        self.softvla_step = softvla_step

        # ── checkpoint paths ────────────────────────────────────────────────
        ckpt_dir = os.path.join(
            "policy", "Soft-VLA", "checkpoints",
            train_config_name, model_name, str(checkpoint_id),
        )
        weight_path = os.path.join(ckpt_dir, "model.safetensors")
        assets_path = os.path.join(ckpt_dir, "assets")

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"SoftVLA checkpoint not found: {weight_path}\n"
                "Expected layout: policy/Soft-VLA/checkpoints/"
                "{train_config_name}/{model_name}/{checkpoint_id}/model.safetensors"
            )

        # ── TrainConfig → transforms ────────────────────────────────────────
        # Fine-tune checkpoints are often saved under "*_ft" roots while the
        # registered OpenPI TrainConfig remains the base name.
        resolved_train_config_name = train_config_name
        try:
            train_config = _config.get_config(resolved_train_config_name)
        except ValueError:
            if train_config_name.endswith("_ft"):
                resolved_train_config_name = train_config_name[: -len("_ft")]
                logger.warning(
                    "TrainConfig '%s' not registered; falling back to '%s' for transforms.",
                    train_config_name,
                    resolved_train_config_name,
                )
                train_config = _config.get_config(resolved_train_config_name)
            else:
                raise
        pi0_cfg = train_config.model  # Pi0Config instance

        # Build SoftVLAConfig whose architecture fields mirror the Pi0Config.
        # num_robots is inferred from the checkpoint weight shape so that
        # fine-tuned checkpoints with an expanded SoftPromptHub load correctly.
        # Use safe_open to read only the tensor shape (no full data load).
        _emb_key = "soft_prompt_hub.embedding.weight"
        with safetensors.torch.safe_open(weight_path, framework="pt", device="cpu") as _f:
            _num_robots = (
                _f.get_slice(_emb_key).get_shape()[0]
                if _emb_key in _f.keys()
                else SoftVLAConfig().num_robots
            )
        logger.info("SoftPromptHub num_robots inferred from checkpoint: %d", _num_robots)

        softvla_cfg = SoftVLAConfig(
            num_robots=_num_robots,
            action_dim=pi0_cfg.action_dim,
            action_horizon=pi0_cfg.action_horizon,
            max_token_len=pi0_cfg.max_token_len,
            pi05=True,
            paligemma_variant=pi0_cfg.paligemma_variant,
            action_expert_variant=pi0_cfg.action_expert_variant,
        )

        # ── load SoftVLAPytorch weights ─────────────────────────────────────
        logger.info("Loading SoftVLAPytorch from %s", weight_path)
        raw_model = SoftVLAPytorch(softvla_cfg)
        safetensors.torch.load_model(raw_model, weight_path, strict=False)
        raw_model.to(torch.bfloat16)
        logger.info("SoftVLAPytorch loaded successfully.")

        # ── wrap for domain_ids injection ───────────────────────────────────
        wrapped = _SoftVLAInferenceWrapper(raw_model, domain_id=domain_id, num_steps=num_denoise_steps)

        # ── norm_stats from checkpoint assets ───────────────────────────────
        entries = [e for e in os.listdir(assets_path) if not e.startswith(".")]
        if not entries:
            raise FileNotFoundError(f"No asset subdirectory found in {assets_path}")
        assets_id = entries[0]
        norm_stats = _normalize.load(os.path.join(assets_path, assets_id))
        # Older checkpoints saved with key "action" (singular); the transforms
        # pipeline uses "actions" (plural).  Rename to avoid Unnormalize strict error.
        if "action" in norm_stats and "actions" not in norm_stats:
            norm_stats = dict(norm_stats)
            norm_stats["actions"] = norm_stats.pop("action")

        # ── data / model transforms ─────────────────────────────────────────
        # Create transforms without loading norm_stats again (assets dir may not
        # exist at the standard location configured in TrainConfig.assets_dirs).
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = _policy.Policy(
            wrapped,
            transforms=[
                _transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
                # Must be last: convert uint8 images to float32 before Policy.infer
                # batches them into torch tensors.  Avoids the torch.uint8 path in
                # Observation.from_dict which incorrectly permutes BHWC→BCHW,
                # causing a shape mismatch in SiGLIP's Conv2d (expects NCHW).
                _CastImagesToFloat(),
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            is_pytorch=True,
            pytorch_device=device,
        )

        self.observation_window: dict | None = None
        self.instruction: str | None = None

    # ── public API (mirrors PI0 in pi05/pi_model.py) ───────────────────────

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction
        logger.info("Instruction set: %s", instruction)

    def update_observation_window(self, img_arr: list, state: np.ndarray) -> None:
        """Update the observation buffer with the latest camera images and end-effector state.

        Args:
            img_arr: List of three HWC uint8 numpy arrays:
                     [head/front camera, right wrist camera, left wrist camera]
            state:   1-D float array. For Soft-VLA right-only EE deployment this
                     is 10-D
                     ``[right_xyz(3), right_rot6d(6), right_gripper(1)]``.
                     Other state representations are also accepted; OpenPI
                     transforms zero-pad to the model action_dim.
        """
        def _to_chw_uint8(img: np.ndarray) -> np.ndarray:
            # RoboTwinEEInputs._convert_image 가 [C, H, W] uint8 을 기대함.
            # (einops.rearrange "c h w -> h w c" 로 HWC로 변환 후 ResizeImages 전달)
            # → 여기서는 단순히 HWC → CHW 변환만 하고 uint8 그대로 유지한다.
            arr = np.asarray(img)

            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]

            if arr.ndim != 3:
                return arr

            # HWC → CHW (이미 CHW이면 그대로)
            if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))

            # float → uint8 변환이 필요하면 수행 (RoboTwin은 보통 uint8 반환)
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

            return arr  # CHW uint8

        img_front = _to_chw_uint8(img_arr[0])
        img_right = _to_chw_uint8(img_arr[1])
        img_left  = _to_chw_uint8(img_arr[2])

        self.observation_window = {
            "state": np.asarray(state, dtype=np.float32),
            "images": {
                "cam_high":        img_front,
                "cam_left_wrist":  img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self) -> np.ndarray:
        """Run inference and return the predicted action chunk.

        Returns:
            np.ndarray of shape (action_horizon, action_dim).
        """
        assert self.observation_window is not None, (
            "Call update_observation_window() before get_action()."
        )
        return self.policy.infer(self.observation_window)["actions"]

    def reset_observation_windows(self) -> None:
        """Clear the observation buffer and instruction at the start of each episode."""
        self.instruction = None
        self.observation_window = None
        logger.info("Observation window and instruction cleared.")

    # ── dual-env server protocol helpers ──────────────────────────────────────

    def reset_model(self) -> None:
        """Alias for reset_observation_windows — called via socket by eval_policy_client."""
        self.reset_observation_windows()

    def get_softvla_step(self) -> int:
        """Return the action chunk execution length (used by server protocol)."""
        return self.softvla_step

    def eval_step(self, obs: dict) -> dict:
        """Single eval step for dual-env socket mode.

        Args:
            obs: dict with keys
                 ``img_arr``    – list of three HWC uint8 ndarrays
                 ``state``      – 1-D float ndarray (10-D right-only EE state)
                 ``instruction``– language instruction string

        Returns:
            dict ``{"actions": ndarray (action_horizon, action_dim),
                    "softvla_step": int}``
        """
        if self.observation_window is None:
            self.set_language(obs["instruction"])
        self.update_observation_window(obs["img_arr"], obs["state"])
        return {"actions": self.get_action(), "softvla_step": self.softvla_step}
