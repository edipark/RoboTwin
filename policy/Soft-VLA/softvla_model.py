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
# This file lives at  policy/Soft-VLA/softvla_model.py.
# The RoboTwin runner sets cwd to the RoboTwin root, so
#   policy/Soft-VLA/src  →  {Soft-VLA repo root}/src
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_POLICY_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from softvla.models.softvla_config import SoftVLAConfig          # noqa: E402
from softvla.models.softvla_pytorch import SoftVLAPytorch        # noqa: E402
from openpi.policies import policy as _policy                    # noqa: E402
from openpi.training import checkpoints as _checkpoints          # noqa: E402
from openpi.training import config as _config                    # noqa: E402
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
        model_name:        Experiment / checkpoint name (directory under train_config_name).
        checkpoint_id:     Checkpoint step number (sub-directory under model_name).
        softvla_step:      Number of flow-matching denoising steps (action execution length).
        domain_id:         Index into SoftPromptHub; selects the embodiment-specific soft prompt.
                           Default 0 works when only one embodiment is registered.
    """

    def __init__(
        self,
        train_config_name: str,
        model_name: str,
        checkpoint_id: int,
        softvla_step: int,
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
        train_config = _config.get_config(train_config_name)
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
        safetensors.torch.load_model(raw_model, weight_path)
        raw_model.to(torch.bfloat16)
        logger.info("SoftVLAPytorch loaded successfully.")

        # ── wrap for domain_ids injection ───────────────────────────────────
        wrapped = _SoftVLAInferenceWrapper(raw_model, domain_id=domain_id, num_steps=softvla_step)

        # ── norm_stats from checkpoint assets ───────────────────────────────
        entries = [e for e in os.listdir(assets_path) if not e.startswith(".")]
        if not entries:
            raise FileNotFoundError(f"No asset subdirectory found in {assets_path}")
        assets_id = entries[0]
        norm_stats = _checkpoints.load_norm_stats(assets_path, assets_id)

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
        """Update the observation buffer with the latest camera images and joint state.

        Args:
            img_arr: List of three HWC uint8 numpy arrays:
                     [head/front camera, right wrist camera, left wrist camera]
            state:   1-D float array of joint positions (e.g. 14-D for bimanual).
        """
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]

        # HWC → CHW as expected by AlohaInputs
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left  = np.transpose(img_left,  (2, 0, 1))

        self.observation_window = {
            "state": state,
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
