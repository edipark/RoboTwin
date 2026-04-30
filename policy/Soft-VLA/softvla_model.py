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
        # SoftVLA-specific fields (num_robots, soft_prompt_length, etc.) retain
        # their defaults, which must match what was used during training.
        softvla_cfg = SoftVLAConfig(
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
        def _to_chw(img: np.ndarray) -> np.ndarray:
            # RoboTwin returns HWC uint8; OpenPI's RoboTwin / Aloha input transforms
            # expect CHW (matching PI05 conventions) and rearrange to HWC internally.
            arr = np.asarray(img)

            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]

            if arr.ndim != 3:
                return arr

            if arr.shape[0] in (1, 3, 4):
                return arr

            if arr.shape[-1] in (1, 3, 4):
                return np.transpose(arr, (2, 0, 1))

            return arr

        img_front = _to_chw(img_arr[0])
        img_right = _to_chw(img_arr[1])
        img_left  = _to_chw(img_arr[2])

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
