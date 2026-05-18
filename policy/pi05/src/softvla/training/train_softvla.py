"""
2-phase Soft-VLA training script (base: DTW-NCE full fine-tuning).

Phase 1:
  - Full fine-tuning of PaliGemma LM (use_lora=False, default, ~2.5B params).
  - LoRA at layer 9 available via --use_lora True (~102K params, phase1_lr=5e-5 권장).
  - DTW-guided soft InfoNCE (l_soft_nce) with topology-aware batching.

Usage:
  Phase 1 (encoder DTW-NCE alignment):
    python -m softvla.training.train_softvla --phase 1 --exp_name softvla_p1_base

  Phase 2 (action generation with soft prompts):
    python -m softvla.training.train_softvla --phase 2 --exp_name softvla_p2 \
        --phase1_encoder_path ./checkpoints/softvla_p1_base/20000

  Multi-GPU (single node, standard Phase 1 settings):
    OMP_NUM_THREADS=8 PYTHONPATH=src .venv/bin/torchrun \
        --standalone --nnodes=1 --nproc_per_node=8 \
        -m softvla.training.train_softvla \
        --phase 1 --exp_name softvla_p1_base \
        --batch_size 128 --phase1_steps 20000 \
        --phase1_lr 2e-5 --phase1_warmup_steps 200 \
        --max_grad_norm 1.0 --nce_temperature 0.1 --nce_max_cdf 0.025 \
        --topology_aware_batching True --topology_group_size 4 --topology_buffer_size 2048 \
        --log_interval 10 --save_interval 10000 \
        --oxe_data_dir /home/yonsei_meat/workspace/datasets \
        --dtw_cdf_path ./assets/softvla/dtw_cdf_sorted_trans.npy \
        --wandb_enabled True --wandb_project Soft-VLA
"""

# ── Suppress TF/XLA CUDA plugin duplicate-registration noise ─────────────────
# Must be set before ANY TensorFlow import (including transitive ones).
# TF_CPP_MIN_LOG_LEVEL: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR (silent)
import os as _os_early
_os_early.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os_early.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
del _os_early

import argparse
import dataclasses
import gc
import logging
import os
import pathlib
import shutil
import threading
import time

import numpy as np
from safetensors import safe_open
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

from softvla.models.softvla_config import SoftVLAConfig
from softvla.models.softvla_pytorch import SoftVLAPytorch
from softvla.models.losses import Phase1Loss, compute_dtw_weights
from softvla.training.config import SoftVLATrainConfig
from softvla.data.oxe_config import NUM_ROBOTS


# ─── Logging ──────────────────────────────────────

def init_logging():
    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
    logging.basicConfig(format=fmt, datefmt="%H:%M:%S", level=logging.INFO, force=True)
    # Suppress noisy internal loggers
    logging.getLogger("be.kuleuven.dtai.distance").setLevel(logging.WARNING)
    logging.getLogger("be.kuleuven.dtai").setLevel(logging.WARNING)
    logging.getLogger("dtaidistance").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("tensorflow_datasets").setLevel(logging.WARNING)
    logging.getLogger("dataset_info").setLevel(logging.WARNING)
    logging.getLogger("reader").setLevel(logging.WARNING)
    logging.getLogger("logging_logger").setLevel(logging.WARNING)
    import os as _os
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # belt-and-suspenders for child procs


# ─── DDP ──────────────────────────────────────────

def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(use_ddp: bool) -> bool:
    return (not use_ddp) or (dist.get_rank() == 0)


# ─── Freeze / Unfreeze helpers ────────────────────

def freeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def setup_phase1_params(model: SoftVLAPytorch, full_finetune: bool = True):
    """Phase 1: freeze everything, then unfreeze trainable parameters.

    Args:
        full_finetune: If True, unfreeze the entire PaliGemma language-model
            (all transformer layers + embedding table).  Gradients to embed_tokens
            are cut off by the detach in forward_phase1, but all attention / FFN
            weights receive gradients normally.  Use phase1_lr ≈ 1e-5.
            If False, unfreeze LoRA adapter weights only (~102K params).
    """
    freeze_module(model)
    if full_finetune:
        # Unfreeze the PaliGemma language model (transformer layers).
        # forward_phase1 detaches prefix_embs before extract_z, so embed_tokens
        # won't receive gradients via that path — but attention/FFN weights will.
        lm = model.pi0.paligemma_with_expert.paligemma.model.language_model
        unfreeze_module(lm)
    else:
        # LoRA: re-enable adapter weights by name (requires_grad was zeroed by
        # freeze_module above, so name-based detection is the correct method).
        lora_params = model.get_lora_params() if hasattr(model, "get_lora_params") else []
        for p in lora_params:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    mode = "full fine-tuning (language model)" if full_finetune else "LoRA only"
    logging.info(f"Phase 1 [DTW-NCE]: {trainable:,} / {total:,} trainable parameters ({mode})")


def setup_phase2_params(model: SoftVLAPytorch, unfreeze_expert: bool = False):
    """Phase 2: train soft_prompts + action_proj, optionally expert. Freeze encoder + adversary."""
    # Freeze everything first
    freeze_module(model)
    # Unfreeze soft prompt hub
    unfreeze_module(model.soft_prompt_hub)
    # Unfreeze action output projection
    unfreeze_module(model.pi0.action_out_proj)
    # Optionally unfreeze Expert Gemma
    if unfreeze_expert:
        unfreeze_module(model.pi0.paligemma_with_expert.gemma_expert)
        unfreeze_module(model.pi0.action_in_proj)
        if model.pi0.pi05:
            unfreeze_module(model.pi0.time_mlp_in)
            unfreeze_module(model.pi0.time_mlp_out)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Phase 2: {trainable:,} / {total:,} trainable parameters (expert_unfrozen={unfreeze_expert})")



# ─── LR schedule ─────────────────────────────────

def cosine_lr(step: int, warmup: int, total: int, peak: float, end: float = 0.0) -> float:
    if step < warmup:
        init_lr = peak / (warmup + 1)
        return init_lr + (peak - init_lr) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return end + (peak - end) * cos


def resolve_model_file(path: str | pathlib.Path) -> pathlib.Path:
    weight_path = pathlib.Path(path)
    return weight_path / "model.safetensors" if weight_path.is_dir() else weight_path


def _sample(items, n: int = 5) -> list[str]:
    return sorted(str(item) for item in items)[:n]


def load_phase1_encoder_weights(
    model: SoftVLAPytorch,
    model_file: pathlib.Path,
    device: torch.device,
    *,
    min_coverage: float = 0.80,
) -> dict[str, object]:
    """Load a Phase 1 checkpoint into ``model.pi0`` with checkpoint-format validation."""
    if not model_file.exists():
        raise FileNotFoundError(f"Phase 1 checkpoint file not found: {model_file}")

    target_state = model.pi0.state_dict()
    target_keys = set(target_state)
    with safe_open(str(model_file), framework="pt", device=str(device)) as handle:
        source_keys = set(handle.keys())
        direct_keys = sorted(key for key in target_keys if key in source_keys)
        prefixed_keys = sorted(key for key in target_keys if f"pi0.{key}" in source_keys)

        if len(prefixed_keys) >= len(direct_keys):
            mode = "softvla_full_prefixed"
            selected = [(key, f"pi0.{key}") for key in prefixed_keys]
        else:
            mode = "pi0_direct"
            selected = [(key, key) for key in direct_keys]

        loaded_state = {}
        shape_mismatches: list[str] = []
        for target_key, source_key in selected:
            tensor = handle.get_tensor(source_key)
            if tuple(tensor.shape) != tuple(target_state[target_key].shape):
                shape_mismatches.append(
                    f"{source_key}: ckpt={tuple(tensor.shape)} target={tuple(target_state[target_key].shape)}"
                )
                continue
            loaded_state[target_key] = tensor

    loaded_count = len(loaded_state)
    coverage = loaded_count / max(len(target_keys), 1)
    if loaded_count == 0:
        raise RuntimeError(
            "Phase 1 checkpoint did not contain any loadable PI0 weights. "
            f"file={model_file}, source_keys={len(source_keys)}, target_keys={len(target_keys)}, "
            f"direct_matches={len(direct_keys)}, prefixed_matches={len(prefixed_keys)}, "
            f"sample_source_keys={_sample(source_keys)}"
        )
    if coverage < min_coverage:
        raise RuntimeError(
            "Phase 1 checkpoint PI0 load coverage is too low. "
            f"file={model_file}, mode={mode}, loaded={loaded_count}/{len(target_keys)} ({coverage:.1%}), "
            f"direct_matches={len(direct_keys)}, prefixed_matches={len(prefixed_keys)}, "
            f"shape_mismatches={_sample(shape_mismatches)}"
        )

    missing, unexpected = model.pi0.load_state_dict(loaded_state, strict=False)
    logging.info(
        "[Phase1Load] Loaded %d/%d PI0 tensors from %s (mode=%s, coverage=%.1f%%, "
        "direct_matches=%d, prefixed_matches=%d, missing=%d, unexpected=%d, shape_mismatches=%d)",
        loaded_count,
        len(target_keys),
        model_file,
        mode,
        coverage * 100,
        len(direct_keys),
        len(prefixed_keys),
        len(missing),
        len(unexpected),
        len(shape_mismatches),
    )
    if missing:
        logging.info("[Phase1Load] Sample missing PI0 keys after load: %s", _sample(missing))
    if unexpected:
        logging.info("[Phase1Load] Sample unexpected PI0 keys after load: %s", _sample(unexpected))
    if shape_mismatches:
        logging.warning("[Phase1Load] Sample shape mismatches skipped: %s", _sample(shape_mismatches))

    return {
        "mode": mode,
        "loaded_count": loaded_count,
        "target_count": len(target_keys),
        "coverage": coverage,
        "direct_matches": len(direct_keys),
        "prefixed_matches": len(prefixed_keys),
    }


# ─── Checkpoint ──────────────────────────────────

# Global slot for the async checkpoint writer thread.
# Only one checkpoint save runs at a time; a new one waits for the previous.
_ckpt_thread: threading.Thread | None = None


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: pathlib.Path,
    *,
    blocking: bool = False,
    tag: str | None = None,
):
    """Save checkpoint asynchronously in a background thread.

    The model state_dict is collected on the calling thread (fast, ~0ms)
    so the saved weights are a consistent snapshot of *this* step.
    The actual NFS write (~6s for 7.2 GB) runs in a daemon thread so the
    training loop continues immediately.

    Args:
        blocking: If True, wait for the write to finish before returning
                  (used for the final checkpoint at end of training).
        tag:      If set, saves to ``ckpt_dir / tag`` instead of
                  ``ckpt_dir / str(step)`` (e.g. tag="best").
    """
    global _ckpt_thread

    # Wait for previous save to finish before starting a new one.
    # This prevents overlapping writes to the same directory.
    if _ckpt_thread is not None and _ckpt_thread.is_alive():
        logging.info("[ckpt] Waiting for previous async save to finish …")
        _ckpt_thread.join()

    model_raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Collect state dicts on the calling thread for a consistent snapshot.
    # .cpu() copies tensors off GPU so the writer thread doesn't need CUDA.
    model_sd = {k: v.cpu() for k, v in model_raw.state_dict().items()}
    optim_sd = optimizer.state_dict()   # already on CPU for AdamW
    meta = {"global_step": step, "timestamp": time.time()}

    dir_name = tag if tag is not None else str(step)

    def _write():
        tmp = ckpt_dir / f"tmp_{dir_name}"
        final = ckpt_dir / dir_name
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(model_sd, str(tmp / "model.safetensors"))
        torch.save(optim_sd, tmp / "optimizer.pt")
        torch.save(meta, tmp / "metadata.pt")
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        if tag is not None:
            logging.info("[ckpt] Saved best checkpoint at step %d → %s", step, final)
        else:
            logging.info("[ckpt] Saved checkpoint at step %d → %s", step, final)

    _ckpt_thread = threading.Thread(target=_write, daemon=True, name=f"ckpt-{dir_name}")
    _ckpt_thread.start()
    if blocking:
        _ckpt_thread.join()


def load_checkpoint(model, optimizer, ckpt_dir: pathlib.Path, device):
    steps = [int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    latest = max(steps)
    d = ckpt_dir / str(latest)

    model_raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.load_model(model_raw, d / "model.safetensors", device=str(device))
    optimizer.load_state_dict(torch.load(d / "optimizer.pt", map_location=device, weights_only=False))
    meta = torch.load(d / "metadata.pt", map_location=device, weights_only=False)
    logging.info(f"Resumed from step {meta['global_step']}")
    return meta["global_step"]


# ─── Fake data generator (for structure testing) ──

def fake_observation(batch_size: int, action_dim: int, device: torch.device):
    """Generate a minimal fake observation matching OpenPI's Observation structure."""
    from openpi.models.model import Observation
    from openpi.shared.array_typing import disable_typechecking

    # Images in [B, H, W, C] format as expected by Observation type annotations
    img_shape = (batch_size, 224, 224, 3)
    images = {
        "base_0_rgb": torch.randn(*img_shape, device=device),
        "left_wrist_0_rgb": torch.randn(*img_shape, device=device),
        "right_wrist_0_rgb": torch.randn(*img_shape, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
    }
    state = torch.randn(batch_size, action_dim, device=device)
    # Fake tokenized prompt (batch of token ids + masks)
    seq_len = 16
    tokenized_prompt = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    with disable_typechecking():
        return Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )


def fake_batch(batch_size: int, config: SoftVLAConfig, device: torch.device, phase: int):
    """Generate a complete fake batch for testing."""
    obs = fake_observation(batch_size, config.action_dim, device)
    domain_ids = torch.randint(0, config.num_robots, (batch_size,), device=device)
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim, device=device)
    return obs, actions, domain_ids


# ─── Main training ────────────────────────────────

def train(train_cfg: SoftVLATrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = is_main_process(use_ddp)
    torch.manual_seed(train_cfg.seed + local_rank)
    np.random.seed(train_cfg.seed + local_rank)

    if train_cfg.save_start_step < 0:
        raise ValueError(f"save_start_step must be >= 0, got {train_cfg.save_start_step}")

    # Model config
    model_cfg = SoftVLAConfig(
        num_robots=NUM_ROBOTS,
        lora_rank=train_cfg.lora_rank,
        lora_alpha=train_cfg.lora_alpha,
        nce_temperature=train_cfg.nce_temperature,
        rotation_weight=train_cfg.rotation_weight,
        gripper_weight=train_cfg.gripper_weight,
        translation_only=train_cfg.translation_only,
        nce_max_cdf=train_cfg.nce_max_cdf,
        training_phase=train_cfg.phase,
        dtype=train_cfg.precision,
        pytorch_compile_mode=None,
    )

    # Build model
    model = SoftVLAPytorch(model_cfg).to(device)

    # Load Phase 1 encoder checkpoint if specified (phase1_encoder_path)
    if train_cfg.phase1_encoder_path is not None:
        model_file = resolve_model_file(train_cfg.phase1_encoder_path)
        logging.info(f"Loading Phase 1 encoder weights from {model_file}")
        load_phase1_encoder_weights(model, model_file, device)

    # Setup phase-specific parameter freezing
    phase = train_cfg.phase
    num_steps = train_cfg.phase1_steps if phase == 1 else train_cfg.phase2_steps
    lr = train_cfg.phase1_lr if phase == 1 else train_cfg.phase2_lr
    warmup = train_cfg.phase1_warmup_steps if phase == 1 else train_cfg.phase2_warmup_steps

    # Checkpoint dir + File logging
    ckpt_dir = pathlib.Path(train_cfg.checkpoint_dir) / train_cfg.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _metrics_file = None
    if is_main:
        _log_path = ckpt_dir / "train.log"
        _fh = logging.FileHandler(_log_path, mode="a")
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logging.getLogger().addHandler(_fh)
        logging.info("Log file: %s", _log_path)

        # ── config.json: full training config snapshot ────────────────────
        import json as _json
        _config_path = ckpt_dir / "config.json"
        _config_dict = dataclasses.asdict(train_cfg)
        _config_path.write_text(_json.dumps(_config_dict, indent=2, default=str))
        logging.info("Config saved → %s", _config_path)

        # ── metrics.jsonl: one JSON line per log interval ─────────────────
        _metrics_path = ckpt_dir / "metrics.jsonl"
        _metrics_file = open(_metrics_path, "a", buffering=1)  # line-buffered
        logging.info("Metrics file: %s", _metrics_path)
        logging.info("[ckpt] Checkpoint writes enabled from step >= %d", train_cfg.save_start_step)

    if phase == 1:
        if train_cfg.use_lora:
            # LoRA injection: fixed middle layer (PaliGemma-3B has 18 layers, index 0-17).
            # Layer 9 (50% depth) carries strong semantic representations without
            # being too close to the output, giving LoRA room to reshape embeddings.
            _LORA_TARGET_LAYER = 9
            logging.info("[Phase 1] Injecting LoRA at layer %d", _LORA_TARGET_LAYER)
            model.apply_lora(_LORA_TARGET_LAYER)
        else:
            logging.info("[Phase 1] Full fine-tuning mode (use_lora=False): LoRA skipped")
        # Freeze everything, then unfreeze trainable params (LoRA or full LM).
        setup_phase1_params(model, full_finetune=not train_cfg.use_lora)

        # Phase 1 loss functions
        phase1_loss = Phase1Loss()
    else:
        phase1_loss = None
        setup_phase2_params(model, unfreeze_expert=False)
    # Gradient checkpointing — must call the method (not just set the flag) so that
    # the setting propagates to all sub-models including the PEFT-wrapped language_model.
    # Full fine-tuning (use_lora=False): activations for 18 layers at batch=128 consume
    # ~100 GB/GPU.  Auto-enable gradient checkpointing to keep peak memory under 183 GB
    # (recomputation trades ~2× backward time for ~10× activation memory reduction).
    _gc_enabled = train_cfg.gradient_checkpointing or (phase == 1 and not train_cfg.use_lora)
    if _gc_enabled:
        model.pi0.gradient_checkpointing_enable()
        reason = "config flag" if train_cfg.gradient_checkpointing else "full fine-tuning (auto)"
        logging.info("[Model] Gradient checkpointing enabled (%s)", reason)

    # DDP
    if use_ddp:
        # find_unused_parameters=False: _set_static_graph() (below) supersedes this and
        # automatically handles unused/frozen parameters, making find_unused_parameters=True
        # redundant (PyTorch emits a warning when both are set).
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        # _set_static_graph(): forward/backward graph is identical every iteration.
        # Required for gradient checkpointing + DDP compatibility:
        # - Prevents "marked ready twice" error from reentrant backward hooks.
        # - Automatically detects unused/frozen parameters (replaces find_unused_parameters=True).
        model._set_static_graph()
        logging.info("[DDP] _set_static_graph() enabled (gradient checkpointing + DDP compat)")

    # Optimizer — separate param groups for soft prompts if phase 2
    model_raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if phase == 2:
        prompt_params = list(model_raw.soft_prompt_hub.parameters())
        other_params = [p for p in model_raw.parameters() if p.requires_grad and not any(p is q for q in prompt_params)]
        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": prompt_params, "lr": lr * train_cfg.lr_coef_soft_prompt},
        ]
    else:
        param_groups = [{"params": [p for p in model_raw.parameters() if p.requires_grad], "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    logging.info("[Optimizer] AdamW (weight_decay=1e-4)")

    # Load precomputed DTW CDF for Phase 1 (if provided)
    cdf_sorted_distances = None
    if phase == 1 and train_cfg.dtw_cdf_path is not None:
        cdf_path = pathlib.Path(train_cfg.dtw_cdf_path)
        if cdf_path.exists():
            # Keep on CPU — DTW is computed on CPU in a background thread.
            # compute_dtw_weights() calls .to(device) internally, so GPU training
            # works fine; keeping it here avoids a GPU→CPU copy on every step.
            cdf_sorted_distances = torch.from_numpy(
                np.load(cdf_path).astype(np.float32)
            )  # CPU tensor
            logging.info(f"Loaded DTW CDF ({len(cdf_sorted_distances):,} distances) from {cdf_path}")
        else:
            logging.warning(f"DTW CDF path not found: {cdf_path} — falling back to raw DTW")

    # Resume
    global_step = 0
    if train_cfg.resume:
        global_step = load_checkpoint(model, optimizer, ckpt_dir, device)

    # Wandb
    if is_main and train_cfg.wandb_enabled:
        # wandb.init() can hang indefinitely when the wandb service fails to connect
        # (e.g. after a previous crashed run left a stale service socket, or due to
        # transient network issues). Run it in a daemon thread with a 60-second timeout
        # so a wandb failure never blocks the training loop.
        _wandb_exc: list = []

        def _wandb_init():
            try:
                wandb.init(
                    name=train_cfg.exp_name,
                    project=train_cfg.wandb_project,
                    config=dataclasses.asdict(train_cfg),
                    settings=wandb.Settings(init_timeout=60),
                )
            except Exception as e:
                _wandb_exc.append(e)

        _wandb_thread = threading.Thread(target=_wandb_init, daemon=True)
        _wandb_thread.start()
        _wandb_thread.join(timeout=60)
        if _wandb_thread.is_alive():
            logging.warning("[wandb] init timed out after 60s — continuing without wandb.")
            train_cfg = dataclasses.replace(train_cfg, wandb_enabled=False)
        elif _wandb_exc:
            logging.warning("[wandb] init failed: %s — continuing without wandb.", _wandb_exc[0])
            train_cfg = dataclasses.replace(train_cfg, wandb_enabled=False)
        else:
            logging.info("[wandb] run initialised: %s", wandb.run.url if wandb.run else "N/A")

    # Training loop
    model.train()
    start_time = time.time()
    pbar = tqdm.tqdm(total=num_steps, initial=global_step, desc=f"Phase {phase}", disable=not is_main) if is_main else None
    infos = []
    _history: list[dict] = []  # compact per-log-step record for summary.json

    # Data loader
    data_iter = None
    if not train_cfg.fake_data:
        from softvla.data.oxe_dataloader import create_oxe_data_loader
        from softvla.data.transforms import batch_to_torch

        # OXEDataLoader base queue must hold at least (topology_buffer_size / batch_size)
        # batches so that _draw_from_base() never blocks waiting for the producer.
        # topology refill draws buffer_size/batch_size = 2048/128 = 16 base batches;
        # add 8 extra slots so TF pipeline can stay warm during refill.
        oxe_base_prefetch = max(
            train_cfg.num_prefetch_batches,
            train_cfg.topology_buffer_size // train_cfg.batch_size + 8,
        ) if train_cfg.topology_aware_batching and phase == 1 else train_cfg.num_prefetch_batches
        logging.info("[OXE] base prefetch=%d (topology_buffer=%d, batch=%d)",
                     oxe_base_prefetch, train_cfg.topology_buffer_size, train_cfg.batch_size)

        oxe_loader = create_oxe_data_loader(
            oxe_data_dir=train_cfg.oxe_data_dir,
            action_horizon=model_cfg.action_horizon,
            action_dim=model_cfg.action_dim,
            batch_size=train_cfg.batch_size,
            norm_stats_dir=train_cfg.norm_stats_dir,
            shuffle=True,
            shuffle_buffer=train_cfg.shuffle_buffer,
            # Each rank must get a different data stream so that the 16 GPUs
            # process *distinct* batches.  Without this, all GPUs see the same
            # data (same TF shuffle seed) and the effective global batch size
            # equals batch_size rather than batch_size × world_size.
            seed=train_cfg.seed + (dist.get_rank() if use_ddp else 0),
            num_prefetch_batches=oxe_base_prefetch,
        )
        if phase == 1 and train_cfg.topology_aware_batching:
            from softvla.data.topology_loader import TopologyAwareOXELoader
            # Prefetch queue must hold 2× the buffer so that _refill_buffer()
            # (draw samples + DTW compute, ~10-20 s) never starves the GPU.
            # With 1× capacity the queue can empty during a slow refill because
            # the training loop keeps consuming while the producer is blocked
            # in _refill_buffer().  2× gives ~48 batch headroom (48 × 3 s =
            # 144 s), well above the worst-case refill time.
            _topology_prefetch = 2 * train_cfg.topology_buffer_size // train_cfg.batch_size
            oxe_loader = TopologyAwareOXELoader(
                base_loader=oxe_loader,
                buffer_size=train_cfg.topology_buffer_size,
                batch_size=train_cfg.batch_size,
                group_size=train_cfg.topology_group_size,
                rotation_weight=model_cfg.rotation_weight,
                gripper_weight=model_cfg.gripper_weight,
                translation_only=model_cfg.translation_only,
                prefetch_batches=_topology_prefetch,
                cross_domain_only=train_cfg.cross_domain_only,
            )
            logging.info(
                "TopologyAwareOXELoader enabled: buffer=%d, group_size=%d, prefetch=%d batches, cross_domain_only=%s",
                train_cfg.topology_buffer_size,
                train_cfg.topology_group_size,
                _topology_prefetch,
                train_cfg.cross_domain_only,
            )
        data_iter = iter(oxe_loader)
        logging.info("Real OXE data loader initialised")

    # ── DTW-weights prefetch (Phase 1 only) ───────────────────────────────────
    # compute_dtw_weights(B=128) takes ~48ms on CPU (dtaidistance, parallel).
    # Running it in the main thread stalls GPU dispatch for 48ms every step.
    # Fix: start the computation in a background thread while the *previous*
    # step's GPU work is running, so it's ready when next(data_iter) returns.
    _dtw_result: list = [None]   # [weights_tensor, n_pos_pairs] or [None]
    _dtw_thread: threading.Thread | None = None

    def _compute_dtw_async(actions_for_dtw: torch.Tensor) -> None:
        with torch.no_grad():
            w = compute_dtw_weights(
                actions_for_dtw,
                rotation_weight=model_cfg.rotation_weight,
                gripper_weight=model_cfg.gripper_weight,
                translation_only=model_cfg.translation_only,
                max_cdf=model_cfg.nce_max_cdf,
                cdf_sorted_distances=cdf_sorted_distances,
            )
            n = int((w > 0).float().sum().item() - w.shape[0])
            _dtw_result[0] = (w, max(n, 0) // 2)

    logging.info("[Train] Waiting for first data batch (TF shuffle buffer + topology DTW init) …")
    best_metric: float = float("inf")  # kl_divergence for phase 1, loss for phase 2
    while global_step < num_steps:
        if train_cfg.fake_data:
            obs, actions, domain_ids = fake_batch(
                train_cfg.batch_size, model_cfg, device, phase
            )
        else:
            obs_np, actions_np, domain_ids_np = next(data_iter)
            obs, actions, domain_ids = batch_to_torch(
                obs_np, actions_np, domain_ids_np, device,
                max_token_len=model_cfg.max_token_len,
            )
            # Keep a CPU tensor snapshot for DTW prefetch thread (next step).
            # torch.from_numpy shares memory; .clone() makes it independent so
            # batch_to_torch's non_blocking H2D copy can't race with the thread.
            if phase == 1:
                actions_np_for_dtw_prefetch = torch.from_numpy(
                    actions_np.astype(np.float32)
                ).clone()

        # Batch domain diversity (main process only)
        n_unique_domains = 0
        domain_entropy = 0.0
        if is_main and (phase == 1 or phase == 2):
            with torch.no_grad():
                _ids = domain_ids.cpu()
                n_unique_domains = int(torch.unique(_ids).numel())
                _counts = torch.bincount(_ids, minlength=model_cfg.num_robots).float()
                _probs = _counts / _counts.sum()
                domain_entropy = float(-(_probs * torch.log(_probs + 1e-8)).sum())

        # LR update
        current_lr = cosine_lr(global_step, warmup, num_steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr * (train_cfg.lr_coef_soft_prompt if pg is param_groups[-1] and phase == 2 else 1.0)

        if phase == 1:
            # ── DTW-NCE Training Step ─────────────────────────────────────────────
            model_raw_p1 = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            # Clip all trainable params (works for both LoRA and full fine-tuning).
            trainable_params_p1 = [p for p in model_raw_p1.parameters() if p.requires_grad]

            # 1. Start DTW for THIS batch's actions immediately (overlap with forward).
            #    Previous design started the thread AFTER forward → result was from
            #    the *previous* batch's actions, mismatched with the current z_student.
            if _dtw_thread is not None:
                _dtw_thread.join()   # clean up any leftover thread
            _dtw_result[0] = None
            if not train_cfg.fake_data:
                _dtw_thread = threading.Thread(
                    target=_compute_dtw_async,
                    args=(actions_np_for_dtw_prefetch,),  # THIS batch's actions
                    daemon=True,
                    name="dtw-this-batch",
                )
                _dtw_thread.start()
            else:
                # fake_data: compute synchronously (no prefetch variable set)
                _compute_dtw_async(actions.cpu())

            # 2. Forward pass (encoder) — DTW runs in parallel on CPU
            z_student = model(phase=1, observation=obs)

            # 3. Wait for THIS batch's DTW result (usually done by GPU forward time)
            if _dtw_thread is not None:
                _dtw_thread.join()
            if _dtw_result[0] is None:
                # Fallback: shouldn't happen with real data, guard for safety
                _compute_dtw_async(actions.cpu())
            dtw_weights_cpu, n_pos_pairs = _dtw_result[0]
            _dtw_result[0] = None

            # Cross-domain positive pair statistics — compute on CPU *before*
            # moving dtw_weights to GPU, so we never trigger a CUDA stream
            # synchronisation (dtw_weights.cpu() on a just-transferred GPU
            # tensor forces a full sync → GPU idle drops every step).
            if train_cfg.cross_domain_only and not train_cfg.fake_data:
                pos_mask_np = (dtw_weights_cpu.numpy() > 0)
                same_np = domain_ids_np[:, None] == domain_ids_np[None, :]
                n_xdomain_pos = int((pos_mask_np & ~same_np).sum()) // 2
                n_samedomain_pos = int((pos_mask_np & same_np).sum()) // 2
                total_pos = n_xdomain_pos + n_samedomain_pos
                xdomain_pos_ratio = n_xdomain_pos / max(total_pos, 1)
            elif train_cfg.cross_domain_only:
                # fake_data path: fall back to torch ops on the GPU tensor
                domain_ids_cpu = domain_ids.cpu()
                same = domain_ids_cpu[:, None] == domain_ids_cpu[None, :]
                pos = dtw_weights_cpu > 0
                n_xdomain_pos = int((pos & ~same).sum().item()) // 2
                n_samedomain_pos = int((pos & same).sum().item()) // 2
                total_pos = n_xdomain_pos + n_samedomain_pos
                xdomain_pos_ratio = n_xdomain_pos / max(total_pos, 1)
            else:
                n_xdomain_pos = 0
                n_samedomain_pos = 0
                xdomain_pos_ratio = 0.0

            dtw_weights = dtw_weights_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            l_nce = phase1_loss.l_soft_nce(z_student, dtw_weights, tau=model_cfg.nce_temperature)
            l_nce.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params_p1, max_norm=train_cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 5. Alignment / Uniformity / Target Entropy (no_grad, detached — zero overhead on backward)
            alignment, uniformity = phase1_loss.compute_alignment_uniformity(
                z_student, dtw_weights
            )
            target_entropy = phase1_loss.compute_target_entropy(dtw_weights)

            loss = train_cfg.lambda_nce * l_nce
            if is_main:
                info = {
                    "loss": loss.item(),
                    "loss_nce": l_nce.item(),
                    "kl_divergence": max(0.0, l_nce.item() - target_entropy),
                    "n_pos_pairs": float(n_pos_pairs),
                    "n_xdomain_pos_pairs": float(n_xdomain_pos),
                    "n_samedomain_pos_pairs": float(n_samedomain_pos),
                    "xdomain_pos_ratio": xdomain_pos_ratio,
                    "lr": current_lr,
                    "grad_norm": float(grad_norm),
                    "n_unique_domains": n_unique_domains,
                    "domain_entropy": domain_entropy,
                    "alignment": alignment,
                    "uniformity": uniformity,
                    "target_entropy": target_entropy,
                }
                # Topology loader cross-domain fallback stats
                if train_cfg.topology_aware_batching and hasattr(oxe_loader, "get_xdomain_stats"):
                    topo_stats = oxe_loader.get_xdomain_stats()
                    info.update(topo_stats)
                infos.append(info)
        else:
            # Phase 2: staged expert unfreezing
            if train_cfg.expert_unfreeze_step is not None and global_step == train_cfg.expert_unfreeze_step:
                setup_phase2_params(model_raw, unfreeze_expert=True)

            mse = model(
                phase=2,
                observation=obs,
                actions=actions,
                domain_ids=domain_ids,
            )
            loss = mse.mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=train_cfg.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main:
                # Soft prompt diagnostics (no_grad, cheap — embedding.weight is tiny)
                with torch.no_grad():
                    sp_w = model_raw.soft_prompt_hub.embedding.weight  # [R, P*D]
                    # Mean L2 norm per robot embedding
                    soft_prompt_norm = float(sp_w.norm(dim=1).mean().item())
                    # Pairwise cosine diversity: mean(1 - cos_sim) over all robot pairs
                    sp_normed = torch.nn.functional.normalize(sp_w, dim=1)  # [R, P*D]
                    cos_sim = sp_normed @ sp_normed.T                        # [R, R]
                    R = cos_sim.shape[0]
                    # Exclude diagonal (self-similarity = 1)
                    mask = ~torch.eye(R, dtype=torch.bool, device=sp_w.device)
                    soft_prompt_diversity = float((1.0 - cos_sim[mask]).mean().item())

                info = {
                    "loss": loss.item(),
                    "lr": current_lr,
                    "grad_norm": float(grad_norm),
                    "soft_prompt_norm": soft_prompt_norm,
                    "soft_prompt_diversity": soft_prompt_diversity,
                    "n_unique_domains": n_unique_domains,
                    "domain_entropy": domain_entropy,
                }
                infos.append(info)

        # Logging
        if is_main and global_step % train_cfg.log_interval == 0 and infos:
            elapsed = time.time() - start_time
            avg = {k: np.mean([i[k] for i in infos if k in i]) for k in infos[0]}
            log_str = f"step={global_step} loss={avg['loss']:.4f} lr={avg['lr']:.2e} grad_norm={avg.get('grad_norm', 0):.2f}"
            if phase == 1:
                log_str += (
                    f" L_nce={avg.get('loss_nce', 0):.4f}"
                    f" pos={avg.get('n_pos_pairs', 0):.0f}"
                    f" xpos={avg.get('n_xdomain_pos_pairs', 0):.0f}"
                    f" xratio={avg.get('xdomain_pos_ratio', 0):.2f}"
                    f" dom={avg.get('n_unique_domains', 0):.0f}/{model_cfg.num_robots}"
                    f" H={avg.get('domain_entropy', 0):.2f}"
                    f" align={avg.get('alignment', 0):.3f}"
                    f" unif={avg.get('uniformity', 0):.3f}"
                    f" H_p={avg.get('target_entropy', 0):.3f}"
                    f" KL={avg.get('kl_divergence', 0):.4f}"
                )
                if avg.get("topology_xdomain_fallback", 0) > 0:
                    log_str += f" fb={avg.get('topology_xdomain_fallback', 0):.0f}"
            elif phase == 2:
                log_str += (
                    f" sp_norm={avg.get('soft_prompt_norm', 0):.3f}"
                    f" sp_div={avg.get('soft_prompt_diversity', 0):.3f}"
                    f" dom={avg.get('n_unique_domains', 0):.1f}/{model_cfg.num_robots}"
                )
            log_str += f" t={elapsed:.1f}s"
            logging.info(log_str)
            if train_cfg.wandb_enabled:
                phase_prefix = f"phase{phase}/"
                wandb.log({phase_prefix + k: v for k, v in avg.items()}, step=global_step)
            # Best checkpoint: phase 1 → minimise KL divergence; phase 2 → minimise loss
            _best_key = "kl_divergence" if phase == 1 else "loss"
            _current_metric = avg.get(_best_key, float("inf"))
            if train_cfg.save_best and global_step >= train_cfg.save_start_step and _current_metric < best_metric:
                best_metric = _current_metric
                logging.info(
                    "[best] New best %s=%.4f at step=%d → saving best checkpoint",
                    _best_key, best_metric, global_step,
                )
                if train_cfg.wandb_enabled:
                    wandb.log(
                        {f"phase{phase}/best_{_best_key}": best_metric, f"phase{phase}/best_step": global_step},
                        step=global_step,
                    )
                save_checkpoint(model, optimizer, global_step, ckpt_dir, tag="best")
            _row = {"step": global_step, **{k: round(float(v), 6) for k, v in avg.items()}}
            _history.append(_row)
            if _metrics_file is not None:
                import json as _json
                _metrics_file.write(_json.dumps(_row) + "\n")
            start_time = time.time()
            infos = []

        # Save (async – NFS write runs in background thread)
        global_step += 1
        if is_main and global_step >= train_cfg.save_start_step and global_step % train_cfg.save_interval == 0:
            save_checkpoint(model, optimizer, global_step, ckpt_dir)

        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Final save (blocking: wait for write before exiting)
    if is_main:
        if global_step >= train_cfg.save_start_step:
            save_checkpoint(model, optimizer, global_step, ckpt_dir, blocking=True)
        else:
            logging.info(
                "[ckpt] Skipping final checkpoint because global_step=%d < save_start_step=%d",
                global_step,
                train_cfg.save_start_step,
            )

        import json as _json

        # ── summary.json: final run summary ───────────────────────────────
        _summary = {
            "exp_name": train_cfg.exp_name,
            "phase": phase,
            "total_steps": global_step,
            "config": dataclasses.asdict(train_cfg),
            "final": _history[-1] if _history else {},
        }
        _summary_path = ckpt_dir / "summary.json"
        _summary_path.write_text(_json.dumps(_summary, indent=2, default=str))
        logging.info("Summary saved → %s", _summary_path)

        if _metrics_file is not None:
            _metrics_file.close()

        if train_cfg.wandb_enabled:
            wandb.finish()
    if pbar:
        pbar.close()

    cleanup_ddp()


# ─── CLI ──────────────────────────────────────────

def parse_args() -> SoftVLATrainConfig:
    parser = argparse.ArgumentParser(description="Soft-VLA Training")
    cfg = SoftVLATrainConfig()
    for field in dataclasses.fields(cfg):
        val = getattr(cfg, field.name)
        actual_type = field.type if field.type is not bool else None
        # Resolve string annotations (e.g. "bool", "str | None")
        if isinstance(actual_type, str):
            actual_type = None
        is_bool = (field.type is bool) or (isinstance(field.type, str) and "bool" in field.type)
        if is_bool:
            # Support both --flag (store_true) and --flag True/False
            parser.add_argument(
                f"--{field.name}",
                type=lambda x: x.lower() not in ("false", "0", "no"),
                default=val,
                metavar="BOOL",
                nargs="?",
                const=True,
            )
        elif val is None:
            # Detect int | None or float | None from annotation
            import types as _types
            from typing import get_args as _get_args
            ann_args = _get_args(field.type) if isinstance(field.type, _types.UnionType) else ()
            non_none = [t for t in ann_args if t is not type(None)]
            if non_none and non_none[0] in (int,):
                _type: type = int
            elif non_none and non_none[0] in (float,):
                _type = float
            elif isinstance(field.type, str):
                ann = field.type
                if "int" in ann and "float" not in ann:
                    _type = int
                elif "float" in ann:
                    _type = float
                else:
                    _type = str
            else:
                _type = str
            parser.add_argument(f"--{field.name}", type=_type, default=None)
        else:
            parser.add_argument(f"--{field.name}", type=type(val), default=val)
    args = parser.parse_args()
    return SoftVLATrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cfg)})


def main():
    init_logging()
    cfg = parse_args()
    logging.info(f"Soft-VLA training: phase={cfg.phase}, exp={cfg.exp_name}")
    train(cfg)


if __name__ == "__main__":
    main()
