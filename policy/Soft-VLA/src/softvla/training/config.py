"""Training configuration for Soft-VLA."""

import dataclasses
from typing import Literal


@dataclasses.dataclass
class SoftVLATrainConfig:
    """Training hyperparameters for 2-phase Soft-VLA training."""

    # ── General ──────────────────────────────────────────────────────────────
    exp_name: str = "softvla_default"
    seed: int = 42
    batch_size: int = 256          # per-GPU; global = batch_size × num_GPUs
    log_interval: int = 10
    save_interval: int = 2500
    wandb_enabled: bool = True
    wandb_project: str = "Soft-VLA"

    # ── Checkpoint ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    resume: bool = False
    phase1_encoder_path: str | None = None  # path to Phase 1 trained encoder checkpoint dir (e.g. ./checkpoints/exp/20000)
    save_best: bool = True  # save best/ checkpoint whenever avg loss improves (checked at log_interval)
    save_start_step: int = 1000  # suppress best/periodic/final checkpoint writes before this step; use 0 for old behavior

    # ── Phase selection ───────────────────────────────────────────────────────
    phase: Literal[1, 2] = 1

    # ── Phase 1: DTW-NCE encoder alignment ───────────────────────────────────
    phase1_steps: int = 10_000
    phase1_lr: float = 2e-5        # full FT default; use higher LR (e.g. 5e-5) for LoRA
    phase1_warmup_steps: int = 200
    lambda_nce: float = 1.0

    # LoRA: parameter-efficient alternative to full fine-tuning.
    # True  → inject LoRA at layer 9 (~102K trainable params), use phase1_lr=5e-5.
    # False → unfreeze all PaliGemma LM layers (~2.5B params), use phase1_lr=1e-5.
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32

    # DTW weight computation
    # Path to pre-sorted DTW distances for CDF normalisation.
    # Generate with: python scripts/compute_dtw_stats.py --translation_only
    dtw_cdf_path: str | None = "./assets/softvla/dtw_cdf_sorted_trans.npy"
    translation_only: bool = True  # use xyz only; ignore rotation/gripper in DTW
    rotation_weight: float = 0.1
    gripper_weight: float = 0.1

    # NCE loss hyperparameters
    nce_temperature: float = 0.1
    # CDF cutoff: pairs with normalised DTW > nce_max_cdf are treated as pure negatives.
    # CLASS uses dist_quantile=0.025. Range 0.025~0.10 works well with topology batching.
    nce_max_cdf: float = 0.025

    # Topology-aware batch sampling (Phase 1 only)
    # Assembles each batch so groups of topology_group_size share DTW neighbours,
    # increasing positive pair density without a global distance matrix.
    topology_aware_batching: bool = True
    topology_buffer_size: int = 2048   # rolling buffer size; must be >= 2 × batch_size
    topology_group_size: int = 4       # 1 anchor + (group_size-1) neighbours per group
    cross_domain_only: bool = True     # force neighbours to come from different robot domains

    # ── Phase 2: action generation ────────────────────────────────────────────
    phase2_steps: int = 10_000
    phase2_lr: float = 2e-4
    phase2_warmup_steps: int = 200
    lr_coef_soft_prompt: float = 1.0   # LR multiplier for soft prompt parameters
    expert_unfreeze_step: int | None = None  # step at which Expert Gemma is unfrozen

    # ── Data ──────────────────────────────────────────────────────────────────
    oxe_data_dir: str = "/home/yonsei_meat/workspace/datasets"
    norm_stats_dir: str | None = "./assets/softvla"
    shuffle_buffer: int = 5_000        # per-worker TF shuffle buffer
    fake_data: bool = False
    num_prefetch_batches: int = 8      # background prefetch queue depth

    # ── Optimisation ─────────────────────────────────────────────────────────
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    max_grad_norm: float = 1.0
    # Gradient checkpointing: required at batch=128/GPU (PaliGemma activations
    # ~120 GB/GPU without it). Recomputes activations during backward (~2× forward
    # time) but keeps peak memory ~10 GB/GPU.
    gradient_checkpointing: bool = True



