"""Soft-VLA configuration."""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class SoftVLAConfig:
    """Configuration for Soft-VLA model wrapping PI0Pytorch.

    Phase 1 trains the encoder to produce embodiment-agnostic Z via DTW-NCE.
    Phase 2 freezes the encoder and trains soft prompts + action head via flow matching.
    """

    # --- Embodiment ---
    num_robots: int = 8

    # --- Encoder (Phase 1) ---
    # Width of PaliGemma LM hidden states (Z dimension before pooling)
    vlm_hidden_dim: int = 2048

    # --- LoRA (Phase 1 selective injection) ---
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # --- DTW-NCE ---
    nce_temperature: float = 0.1
    # Down-weight rotation channels (indices 3:9) in DTW distance computation.
    rotation_weight: float = 0.1
    # Scale factor for gripper channel (index 9). 0.0 to ignore gripper.
    gripper_weight: float = 0.1
    # If True, use only xyz translation (channels 0:3) for DTW; ignore rotation/gripper.
    translation_only: bool = True
    # CDF cutoff: pairs with CDF-normalised distance > max_cdf receive zero weight
    # (treated as pure negatives). Only applies when cdf_sorted_distances is loaded.
    nce_max_cdf: float = 0.025

    # --- Soft Prompt (Phase 2) ---
    soft_prompt_length: int = 32
    # Width of Expert Gemma hidden states
    expert_hidden_dim: int = 1024

    # --- Training phase ---
    training_phase: Literal[1, 2] = 1

    # --- PI0 backbone (set externally) ---
    # These mirror Pi0Config fields and are passed through when constructing PI0Pytorch.
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 16
    max_token_len: int = 200
    pi05: bool = True
    pytorch_compile_mode: str | None = None  # disable compile for training flexibility
