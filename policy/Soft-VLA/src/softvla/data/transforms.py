"""Batch conversion utilities for Soft-VLA training.

Converts numpy observation dicts produced by OXEDataLoader into PyTorch tensors
and wraps them in the OpenPI Observation dataclass expected by SoftVLAPytorch.
"""

from __future__ import annotations

import numpy as np
import torch

from openpi.models.model import Observation
from openpi.shared.array_typing import disable_typechecking


def batch_to_torch(
    obs_np: dict,
    actions_np: np.ndarray,
    domain_ids_np: np.ndarray,
    device: torch.device,
    max_token_len: int,
) -> tuple[Observation, torch.Tensor, torch.Tensor]:
    """Convert a numpy batch from OXEDataLoader to PyTorch tensors.

    Args:
        obs_np:        Observation dict with numpy arrays
                       (keys: images, image_masks, state,
                        tokenized_prompt, tokenized_prompt_mask).
        actions_np:    [B, H, action_dim]  float32.
        domain_ids_np: [B]  int32.
        device:        Target torch device.
        max_token_len: Expected prompt token length (for shape checks).

    Returns:
        (observation, actions_tensor, domain_ids_tensor)
    """
    # ── Images ────────────────────────────────────────────────────────────
    images = {
        k: torch.from_numpy(v.astype(np.float32)).to(device, non_blocking=True)
        for k, v in obs_np["images"].items()
    }
    image_masks = {
        k: torch.from_numpy(v.astype(bool)).to(device, non_blocking=True)
        for k, v in obs_np["image_masks"].items()
    }

    # right_wrist_0_rgb is absent from OXE datasets.  The model's embed_prefix()
    # checks image_masks before using each camera, so a zero-filled tensor is safe.
    # Create it directly on GPU (no CPU allocation, no H2D transfer).
    if "right_wrist_0_rgb" not in images:
        ref_shape = next(iter(images.values())).shape  # [B, H, W, 3]
        images["right_wrist_0_rgb"] = torch.zeros(ref_shape, dtype=torch.float32, device=device)
        image_masks["right_wrist_0_rgb"] = torch.zeros(ref_shape[0], dtype=torch.bool, device=device)

    # ── State ─────────────────────────────────────────────────────────────
    state = torch.from_numpy(obs_np["state"].astype(np.float32)).to(device, non_blocking=True)

    # ── Language tokens ───────────────────────────────────────────────────
    tokenized_prompt = torch.from_numpy(obs_np["tokenized_prompt"].astype(np.int64)).to(device, non_blocking=True)
    tokenized_prompt_mask = torch.from_numpy(obs_np["tokenized_prompt_mask"].astype(bool)).to(device, non_blocking=True)

    # ── Observation ───────────────────────────────────────────────────────
    with disable_typechecking():
        observation = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    # ── Actions & domain IDs ──────────────────────────────────────────────
    actions = torch.from_numpy(actions_np.astype(np.float32)).to(device, non_blocking=True)
    domain_ids = torch.from_numpy(domain_ids_np.astype(np.int64)).to(device, non_blocking=True)

    return observation, actions, domain_ids
