"""Fine-tune Soft-VLA on RoboTwin task data (Phase 2: soft-prompt + action head).

This script is invoked by policy/Soft-VLA/finetune.sh.
It runs from the policy/Soft-VLA/ directory.

Phase 2 freezes everything except:
    - model.soft_prompt_hub          (embodiment-specific soft prompts)
    - model.pi0.action_out_proj      (action output projection)
  Optionally at --unfreeze_expert_step:
    - model.pi0.paligemma_with_expert.gemma_expert
    - model.pi0.action_in_proj
    - model.pi0.time_mlp_in / time_mlp_out  (Pi05 mode)

Checkpoint layout (compatible with policy/Soft-VLA/eval.sh):
    checkpoints/softvla_robotwin_ft/<exp_name>/<step>/
        model.safetensors
        optimizer.pt
        metadata.pt
        assets/
            robotwin/
                norm_stats.json
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import pathlib
import json
import shutil
import threading
import time

import numpy as np
import tqdm
import torch
import torch.multiprocessing as torch_mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ── Resolve Soft-VLA src ──────────────────────────────────────────────────────
_POLICY_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROBOTWIN_ROOT = os.path.dirname(os.path.dirname(_POLICY_DIR))
_SRC_DIR       = os.path.join(_POLICY_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── Local imports (scripts/ on path) ─────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from robotwin_dataset import RoboTwinDataset
from compute_norm_stats import compute_and_save as compute_norm_stats

from openpi.shared import normalize as _normalize
from openpi.shared.array_typing import disable_typechecking
from openpi.models.model import Observation

from softvla.models.softvla_pytorch import SoftVLAPytorch
from softvla.models.softvla_config import SoftVLAConfig

import safetensors.torch

# ── W&B (optional dependency) ────────────────────────────────────────────────
try:
    import wandb as _wandb
except ImportError:
    _wandb = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Local training helpers (mirrors train_softvla.py) ───────────────────────
_ckpt_thread: threading.Thread | None = None


def freeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def setup_phase2_params(model: SoftVLAPytorch, unfreeze_expert: bool = False) -> None:
    """Phase 2: train soft prompts + action projection, optional expert."""
    freeze_module(model)
    unfreeze_module(model.soft_prompt_hub)
    unfreeze_module(model.pi0.action_out_proj)
    if unfreeze_expert:
        unfreeze_module(model.pi0.paligemma_with_expert.gemma_expert)
        unfreeze_module(model.pi0.action_in_proj)
        if model.pi0.pi05:
            unfreeze_module(model.pi0.time_mlp_in)
            unfreeze_module(model.pi0.time_mlp_out)


def cosine_lr(step: int, warmup: int, total: int, peak: float, end: float = 0.0) -> float:
    if step < warmup:
        init_lr = peak / (warmup + 1)
        return init_lr + (peak - init_lr) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return end + (peak - end) * cos


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: pathlib.Path,
    *,
    norm_stats: dict | None = None,
    blocking: bool = False,
) -> None:
    """Async checkpoint writer compatible with Soft-VLA checkpoint format."""
    global _ckpt_thread

    if _ckpt_thread is not None and _ckpt_thread.is_alive():
        _ckpt_thread.join()

    model_raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model_sd = {k: v.cpu() for k, v in model_raw.state_dict().items()}
    optim_sd = optimizer.state_dict()
    meta = {"global_step": step, "timestamp": time.time()}

    def _write():
        tmp = ckpt_dir / f"tmp_{step}"
        final = ckpt_dir / str(step)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(model_sd, str(tmp / "model.safetensors"))
        torch.save(optim_sd, tmp / "optimizer.pt")
        torch.save(meta, tmp / "metadata.pt")
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        # Write norm_stats AFTER rename so it is never deleted by rmtree(final).
        if norm_stats is not None:
            _save_norm_stats_with_ckpt(norm_stats, final)

    _ckpt_thread = threading.Thread(target=_write, daemon=True, name=f"ckpt-{step}")
    _ckpt_thread.start()
    if blocking:
        _ckpt_thread.join()


# ─── Multi-GPU / DDP helpers ──────────────────────────────────────────────────

def _maybe_relaunch_torchrun() -> None:
    """If --gpu_ids has >1 GPU and we are not inside a torchrun context,
    relaunch the current script under ``torchrun`` with the appropriate
    CUDA_VISIBLE_DEVICES, then exec-replace this process (no return).

    For a single GPU (--gpu_ids 0) the function just sets
    CUDA_VISIBLE_DEVICES and returns normally.
    """
    if "LOCAL_RANK" in os.environ:
        return  # already inside torchrun

    argv = sys.argv[1:]
    gpu_ids: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--gpu_ids":
            j = i + 1
            while j < len(argv) and not argv[j].startswith("-"):
                gpu_ids.append(argv[j])
                j += 1
        i += 1

    if len(gpu_ids) == 0:
        return  # no --gpu_ids given, proceed with default CUDA device

    if len(gpu_ids) == 1:
        # Single GPU: set visibility and continue normally
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        log.info("Single GPU selected: CUDA_VISIBLE_DEVICES=%s", gpu_ids[0])
        return

    # Multiple GPUs: relaunch under torchrun
    gpu_str = ",".join(gpu_ids)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1",
        f"--nproc_per_node={len(gpu_ids)}",
        os.path.abspath(__file__),
        *argv,
    ]
    log.info(
        "Multi-GPU: relaunching under torchrun "
        "(CUDA_VISIBLE_DEVICES=%s, nproc_per_node=%d)",
        gpu_str, len(gpu_ids),
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str}
    os.execvpe(cmd[0], cmd, env)
    # os.execvpe replaces the current process — we never reach here


def _setup_ddp(ddp_backend: str | None = None) -> tuple[bool, int, torch.device]:
    """Initialise DDP if WORLD_SIZE > 1 (i.e. we were launched by torchrun).

    Mirrors train_softvla.py setup_ddp():
    - backend auto-selected (nccl on GPU, gloo on CPU)
    - device_id is NOT passed to init_process_group to avoid NCCL SIGSEGV on
      some driver/IB-stack combinations where the eager device-pinning path
      triggers a segfault before the communicator is fully ready.

    Returns:
        use_ddp    – True when running under DDP
        local_rank – this process's GPU index within the node
        device     – the torch.device to use
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if use_ddp and not dist.is_initialized():
        backend = ddp_backend or ("nccl" if torch.cuda.is_available() else "gloo")
        log.info(
            "Initialising process group: backend=%s rank=%s world_size=%s local_rank=%s",
            backend,
            os.environ.get("RANK", "?"),
            os.environ.get("WORLD_SIZE", "?"),
            local_rank,
        )
        dist.init_process_group(backend=backend, init_method="env://")
        log.info("Process group initialised.")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _load_with_hub_expansion(
    model,
    src_weight: str,
    pretrained_num_robots: int,
) -> None:
    """Load *src_weight* into *model*, safely expanding ``soft_prompt_hub``.

    If the pre-trained embedding table has fewer rows than the model's
    embedding table (because ``domain_id >= pretrained_num_robots``),
    this function:
      1. Copies the pre-trained rows into the matching positions.
      2. Leaves new rows at their randomly-initialised values.
      3. Loads all other weights normally (strict=False).

    Args:
        model: The ``SoftVLAPytorch`` instance (already on the target device).
        src_weight: Path to the pre-trained ``model.safetensors``.
        pretrained_num_robots: ``num_robots`` of the *pre-trained* model.
            Used to detect a shape mismatch on the embedding weight.
    """
    state_dict = safetensors.torch.load_file(src_weight)
    emb_key = "soft_prompt_hub.embedding.weight"
    if emb_key in state_dict:
        pretrained_emb = state_dict[emb_key]           # [N_pretrained, D]
        model_num_robots = model.soft_prompt_hub.embedding.num_embeddings
        if pretrained_emb.shape[0] < model_num_robots:
            # Build expanded weight tensor: start from the model's random init
            # so new rows are already properly initialised.
            expanded = model.soft_prompt_hub.embedding.weight.data.detach().clone()
            n = pretrained_emb.shape[0]
            expanded[:n] = pretrained_emb.to(expanded.dtype).to(expanded.device)
            state_dict[emb_key] = expanded
            log.info(
                "SoftPromptHub: expanded embedding %d → %d robots "
                "(rows 0-%d copied from pre-trained; rows %d-%d randomly initialised)",
                pretrained_emb.shape[0], model_num_robots,
                n - 1, n, model_num_robots - 1,
            )
    model.load_state_dict(state_dict, strict=False)


# ─── Observation builder ──────────────────────────────────────────────────────

def batch_to_observation(
    batch: dict,
    device: torch.device,
) -> tuple[Observation, torch.Tensor, torch.Tensor]:
    """Convert a DataLoader batch dict into (Observation, actions, domain_ids).

    Image key mapping:
        cam_high        → base_0_rgb        (head/overhead camera)
        cam_left_wrist  → left_wrist_0_rgb
        cam_right_wrist → right_wrist_0_rgb
    """
    B = batch["cam_high"].shape[0]

    def to_dev(t: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        return t.to(device=device, dtype=dtype, non_blocking=True)

    images = {
        "base_0_rgb":        to_dev(batch["cam_high"]),
        "left_wrist_0_rgb":  to_dev(batch["cam_left_wrist"]),
        "right_wrist_0_rgb": to_dev(batch["cam_right_wrist"]),
    }
    image_masks = {k: torch.ones(B, dtype=torch.bool, device=device) for k in images}

    state               = to_dev(batch["state"])                          # [B, D]
    tokenized_prompt    = batch["tokenized_prompt"].to(device=device, dtype=torch.long)
    tokenized_prompt_mask = batch["tokenized_prompt_mask"].to(device=device, dtype=torch.bool)

    with disable_typechecking():
        obs = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    actions    = to_dev(batch["action"])                   # [B, H, D]
    domain_ids = batch["domain_id"].to(device=device, dtype=torch.long)  # [B]

    return obs, actions, domain_ids


# ─── Norm-stats helpers ───────────────────────────────────────────────────────

def _resolve_norm_stats(args) -> tuple[dict, str]:
    """Return (norm_stats_dict, norm_stats_dir).

    Order of precedence:
      1. --norm_stats_dir  (user-provided, must already exist)
      2. assets/robotwin/<task>-<config>-<N>/  (auto-computed once, then cached)
    """
    if args.norm_stats_dir:
        d = args.norm_stats_dir if os.path.isabs(args.norm_stats_dir) else \
            os.path.join(_POLICY_DIR, args.norm_stats_dir)
        stats = _normalize.load(d)
        log.info("Loaded norm stats from %s", d)
        return stats, d

    auto_dir = os.path.join(
        _POLICY_DIR,
        "assets", "robotwin",
        f"{args.task_name}-{args.task_config}-{args.expert_data_num}",
    )
    if not os.path.isfile(os.path.join(auto_dir, "norm_stats.json")):
        processed_dir = os.path.join(
            _POLICY_DIR,
            "processed_data",
            f"{args.task_name}-{args.task_config}-{args.expert_data_num}",
        )
        log.info("Computing norm stats from %s → %s", processed_dir, auto_dir)
        compute_norm_stats(processed_dir, auto_dir, action_horizon=args.action_horizon)

    stats = _normalize.load(auto_dir)
    log.info("Loaded norm stats from %s", auto_dir)
    return stats, auto_dir


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_norm_stats_with_ckpt(
    norm_stats: dict,
    ckpt_step_dir: pathlib.Path,
    asset_id: str = "robotwin",
):
    """Save norm_stats alongside a checkpoint for eval.sh compatibility.

    eval.sh expects:
        <ckpt_step_dir>/assets/<asset_id>/norm_stats.json
    """
    assets_dir = ckpt_step_dir / "assets" / asset_id
    assets_dir.mkdir(parents=True, exist_ok=True)
    _normalize.save(str(assets_dir), norm_stats)
    log.info("[ckpt] Saved norm stats → %s", assets_dir)


def _resolve_src_ckpt(src_ckpt_dir: str) -> str:
    """Resolve src checkpoint dir: relative → anchored at RoboTwin root."""
    if os.path.isabs(src_ckpt_dir):
        return src_ckpt_dir
    # Try relative to RoboTwin root first (eval.sh convention)
    candidate = os.path.join(_ROBOTWIN_ROOT, src_ckpt_dir)
    if os.path.isdir(candidate):
        return candidate
    # Fall back: relative to policy/Soft-VLA/
    return os.path.join(_POLICY_DIR, src_ckpt_dir)


# ─── W&B helpers ─────────────────────────────────────────────────────────────

def _wandb_init(args):
    """Initialize a W&B run.  Emits a warning (no crash) if wandb is not installed."""
    if _wandb is None:
        log.warning(
            "--wandb_enabled set but 'wandb' is not installed.  "
            "Run: pip install wandb"
        )
        return
    _wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or args.exp_name,
        config=vars(args),
        resume="allow",
        id=args.wandb_run_id or None,
    )
    log.info(
        "W&B run initialised: project=%s  name=%s",
        args.wandb_project,
        args.wandb_run_name or args.exp_name,
    )


def _wandb_log(metrics: dict, step: int):
    """Log metrics dict aligned on the global training step.

    Uses the *step* kwarg (not a dict key) so all metrics share the same
    x-axis without creating a spurious 'step' metric series.
    """
    if _wandb is not None and _wandb.run is not None:
        _wandb.log(metrics, step=step)


def _wandb_finish():
    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()


# ─── GPU memory helper ────────────────────────────────────────────────────────

def _gpu_mem_stats() -> dict:
    """Return GPU memory metrics (GB).  Returns empty dict if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return {}
    alloc  = torch.cuda.memory_allocated()  / 1e9
    reserv = torch.cuda.memory_reserved()   / 1e9
    peak   = torch.cuda.max_memory_allocated() / 1e9
    return {
        "system/gpu_mem_alloc_gb":   alloc,
        "system/gpu_mem_reserved_gb": reserv,
        "system/gpu_mem_peak_gb":    peak,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Soft-VLA (Phase 2) on a RoboTwin task."
    )
    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--task_name",       required=True)
    p.add_argument("--task_config",     required=True)
    p.add_argument("--expert_data_num", type=int, default=50)
    p.add_argument(
        "--norm_stats_dir",
        default=None,
        help="Pre-computed norm_stats dir.  Auto-computed from processed_data if omitted.",
    )

    # ── Pre-trained model ─────────────────────────────────────────────────
    p.add_argument(
        "--src_ckpt_dir",
        default=None,
        help="Pre-trained checkpoint root dir (contains <step>/model.safetensors). "
             "Mutually exclusive with --phase2_checkpoint.",
    )
    p.add_argument("--src_ckpt_step", type=int, default=None)
    p.add_argument(
        "--phase2_checkpoint",
        default=None,
        help="Direct path to a checkpoint directory containing model.safetensors. "
             "e.g. ./checkpoints/p2_xvla_action_decoder_v1/best. "
             "Mutually exclusive with --src_ckpt_dir/--src_ckpt_step.",
    )

    # ── Model architecture (must match pre-trained model) ─────────────────
    p.add_argument(
        "--action_dim",
        type=int,
        default=32,
        help="Model-side action / state dimension (zero-padded).  Must match pre-trained "
             "model (dtw_fullft_v4_re_phase2 = 32).  The real right-only EE action dim "
             "produced by process_data.py is 10 (xyz + rot6d + gripper); the dataset and "
             "OpenPI transforms zero-pad up to this value.",
    )
    p.add_argument(
        "--action_horizon",
        type=int,
        default=16,
        help="Action chunk length (must match pre-trained model).",
    )
    p.add_argument("--max_token_len", type=int, default=200)
    p.add_argument(
        "--num_robots",
        type=int,
        default=9,
        help="SoftPromptHub capacity (must match pre-trained model).",
    )
    p.add_argument(
        "--soft_prompt_length",
        type=int,
        default=32,
        help="Number of soft-prompt tokens per embodiment.",
    )
    p.add_argument(
        "--domain_id",
        type=int,
        default=8,
        help="Embodiment index for SoftPromptHub (0-based, < num_robots).",
    )

    # ── Training hyper-parameters ─────────────────────────────────────────
    p.add_argument("--exp_name",         required=True)
    p.add_argument("--finetune_steps",   type=int,   default=10_000)
    p.add_argument("--batch_size",       type=int,   default=256)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--warmup_steps",     type=int,   default=200)
    p.add_argument("--save_interval",    type=int,   default=1_000)
    p.add_argument("--log_interval",     type=int,   default=10)
    p.add_argument("--num_workers",      type=int,   default=4)

    # ── 2-stage adaptation (LIBERO-style) ────────────────────────────────
    p.add_argument(
        "--stage1_steps",
        type=int,
        default=None,
        help="Stage 1 step count: train soft_prompt + action_out_proj only (backbone frozen). "
             "When set, overrides --finetune_steps; total = stage1_steps + stage2_steps.",
    )
    p.add_argument(
        "--stage2_steps",
        type=int,
        default=0,
        help="Stage 2 step count: joint training with Expert Gemma + action_in_proj unfrozen. "
             "Only used when --stage1_steps is set.",
    )
    p.add_argument(
        "--stage2_lr",
        type=float,
        default=None,
        help="Peak LR for Stage 2 joint training. Defaults to --lr when not set.",
    )
    p.add_argument("--seed",             type=int,   default=42)

    # ── Expert Gemma unfreezing ────────────────────────────────────────────
    p.add_argument(
        "--unfreeze_expert_step",
        type=int,
        default=None,
        help="Step at which to unfreeze the Expert Gemma and action_in_proj.",
    )

    # ── Multi-GPU ──────────────────────────────────────────────────────────
    p.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        metavar="GPU",
        help="GPU IDs to use (e.g. --gpu_ids 0 1 2 3).  "
             "A single ID sets CUDA_VISIBLE_DEVICES.  "
             "Multiple IDs auto-relaunch the script under torchrun (DDP).",
    )
    p.add_argument(
        "--ddp_backend",
        type=str,
        default=None,
        choices=["nccl", "gloo"],
        help="DDP backend override. Default: nccl on CUDA, gloo on CPU.",
    )

    # ── Misc ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the output directory.",
    )
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before an optimizer step. "
             "Effective batch size = batch_size * gradient_accumulation_steps.",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage at the cost of ~30%% more compute.",
    )
    p.add_argument("--image_size", type=int, default=224)

    # ── W&B ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--wandb_enabled",
        action="store_true",
        help="Enable Weights & Biases logging.  Requires: pip install wandb",
    )
    p.add_argument(
        "--wandb_project",
        default="Soft-VLA_RoboTwin_FineTune",
        help="W&B project name.  Defaults to 'Soft-VLA' to land in the same team project as other training runs.",
    )
    p.add_argument(
        "--wandb_entity",
        default=None,
        help="W&B entity (team name or username).  Set to your team entity so runs appear in the shared team project.",
    )
    p.add_argument(
        "--wandb_run_name",
        default=None,
        help="W&B run display name.  Defaults to --exp_name.",
    )
    p.add_argument(
        "--wandb_run_id",
        default=None,
        help="W&B run ID.  Set to resume logging into an existing run.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ── Multi-GPU: relaunch under torchrun if needed ──────────────────────
    _maybe_relaunch_torchrun()   # may exec-replace this process (no return)
    use_ddp, local_rank, device = _setup_ddp(args.ddp_backend)
    _is_main = (not use_ddp) or (dist.get_rank() == 0)

    if use_ddp:
        log.info(
            "DDP: rank=%d / world=%d  device=%s",
            dist.get_rank(), dist.get_world_size(), device,
        )
    else:
        log.info("Device: %s", device)

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # ── Output checkpoint dir ────────────────────────────────────────────
    # All ranks create the directory (exist_ok=True is safe); no barrier needed.
    ckpt_dir = pathlib.Path(_ROBOTWIN_ROOT) / "policy" / "Soft-VLA" / "checkpoints" / "softvla_robotwin_ft" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _metrics_file = None
    if _is_main:
        _log_path = ckpt_dir / "train.log"
        _fh = logging.FileHandler(_log_path, mode="a")
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logging.getLogger().addHandler(_fh)
        log.info("Log file: %s", _log_path)

        _metrics_path = ckpt_dir / "metrics.jsonl"
        _metrics_file = open(_metrics_path, "a", buffering=1)  # line-buffered
        log.info("Metrics file: %s", _metrics_path)

    # ── Norm stats ───────────────────────────────────────────────────────
    norm_stats, norm_stats_dir = _resolve_norm_stats(args)

    # ── Dataset & DataLoader ─────────────────────────────────────────────
    processed_dir = os.path.join(
        _POLICY_DIR,
        "processed_data",
        f"{args.task_name}-{args.task_config}-{args.expert_data_num}",
    )
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(
            f"Processed data not found: {processed_dir}\n"
            f"Run first:  bash process_data_softvla.sh "
            f"{args.task_name} {args.task_config} {args.expert_data_num}"
        )

    dataset = RoboTwinDataset(
        processed_dir=processed_dir,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        max_token_len=args.max_token_len,
        norm_stats=norm_stats,
        domain_id=args.domain_id,
        image_size=args.image_size,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    # Use 'spawn' start method for DataLoader workers so they don't inherit any
    # CUDA context from the parent process (Linux default 'fork' + CUDA = SIGSEGV).
    # Matches the approach used by train_softvla.py's OXE/Libero data loaders.
    _mp_ctx = torch_mp.get_context("spawn") if args.num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        multiprocessing_context=_mp_ctx,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    if _is_main:
        log.info(
            "Dataset: %d samples | %d batches/epoch (batch=%d, world=%d)",
            len(dataset), len(loader), args.batch_size,
            dist.get_world_size() if use_ddp else 1,
        )

    # ── Build model ───────────────────────────────────────────────────────
    # Auto-expand num_robots if domain_id exceeds the pre-trained capacity.
    effective_num_robots = max(args.num_robots, args.domain_id + 1)
    if effective_num_robots > args.num_robots and _is_main:
        log.info(
            "domain_id=%d >= num_robots=%d: auto-expanding SoftPromptHub to %d slots",
            args.domain_id, args.num_robots, effective_num_robots,
        )

    softvla_cfg = SoftVLAConfig(
        num_robots=effective_num_robots,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        pi05=True,
        soft_prompt_length=args.soft_prompt_length,
        training_phase=2,
    )
    model = SoftVLAPytorch(softvla_cfg).to(device)

    # ── Load pre-trained weights (with hub expansion if needed) ───────────
    # Two mutually exclusive loading modes:
    #   (a) --phase2_checkpoint <dir>  → dir/model.safetensors loaded directly
    #   (b) --src_ckpt_dir + --src_ckpt_step  → <dir>/<step>/model.safetensors (legacy)
    if args.phase2_checkpoint is not None:
        ckpt_path = pathlib.Path(args.phase2_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = pathlib.Path.cwd() / args.phase2_checkpoint
        src_weight = str(ckpt_path / "model.safetensors")
        if not os.path.isfile(src_weight):
            raise FileNotFoundError(
                f"--phase2_checkpoint: model.safetensors not found at {src_weight}"
            )
        if _is_main:
            log.info("Loading pre-trained weights from --phase2_checkpoint: %s", src_weight)
    else:
        if args.src_ckpt_dir is None or args.src_ckpt_step is None:
            raise ValueError(
                "Must provide either --phase2_checkpoint  OR  "
                "both --src_ckpt_dir and --src_ckpt_step."
            )
        src_ckpt_root = _resolve_src_ckpt(args.src_ckpt_dir)
        src_weight = os.path.join(src_ckpt_root, str(args.src_ckpt_step), "model.safetensors")
        if not os.path.isfile(src_weight):
            raise FileNotFoundError(f"Pre-trained model not found: {src_weight}")
        if _is_main:
            log.info("Loading pre-trained weights from %s", src_weight)
    _load_with_hub_expansion(model, src_weight, pretrained_num_robots=args.num_robots)

    # ── Gradient checkpointing (must be set before DDP wrapping) ─────────
    if args.gradient_checkpointing:
        model.pi0.gradient_checkpointing_enable()
        if _is_main:
            log.info("Gradient checkpointing enabled.")

    # ── Phase 2 param setup (freeze backbone, unfreeze prompts+action_out) ─
    setup_phase2_params(model, unfreeze_expert=False)

    # ── Wrap with DDP ─────────────────────────────────────────────────────
    if use_ddp:
        # find_unused_parameters=True is required when:
        #   (a) frozen parameters exist (not all params produce gradients), AND/OR
        #   (b) gradient checkpointing is active inside the model.
        # _set_static_graph() is incompatible with gradient checkpointing because
        # checkpointing recomputes the forward pass during backward, changing the
        # autograd hook call order that static-graph mode expects, which causes
        # "expect_autograd_hooks_ INTERNAL ASSERT FAILED" (reducer.cpp:1633).
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False, # Action expert full finetune
            gradient_as_bucket_view=True,
        )
    model_raw = model.module if use_ddp else model

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in model_raw.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # ── Optional resume ────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        existing = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if existing:
            latest = existing[-1]
            if _is_main:
                log.info("Resuming from %s", latest)
            safetensors.torch.load_model(model_raw, str(latest / "model.safetensors"))
            optimizer.load_state_dict(
                torch.load(latest / "optimizer.pt", map_location=device, weights_only=False)
            )
            meta = torch.load(latest / "metadata.pt", map_location="cpu", weights_only=False)
            start_step = int(meta["global_step"]) + 1
            if _is_main:
                log.info("Resumed at step %d", start_step)

    # ── W&B init (main process only) ─────────────────────────────────────
    if args.wandb_enabled and _is_main:
        _wandb_init(args)

    # ── Log trainable param count (main process only) ────────────────────
    total_params      = sum(p.numel() for p in model_raw.parameters())
    trainable_count   = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
    if _is_main:
        log.info(
            "Parameters: total=%s  trainable=%s  frozen=%s  (%.2f%% trainable)  "
            "num_robots_effective=%d",
            f"{total_params:,}",
            f"{trainable_count:,}",
            f"{total_params - trainable_count:,}",
            100.0 * trainable_count / total_params if total_params else 0,
            effective_num_robots,
        )
        if args.wandb_enabled:
            _wandb_log(
                {
                    "model/total_params":       total_params,
                    "model/trainable_params":   trainable_count,
                    "model/frozen_params":      total_params - trainable_count,
                    "model/num_robots_effective": effective_num_robots,
                },
                step=start_step,
            )
    # Reset peak memory counter before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    step = start_step
    expert_unfrozen = False
    loss_acc = 0.0
    loss_sq_acc = 0.0   # for std computation
    _loader_epoch = 0   # tracks dataset epochs for DistributedSampler

    # ── 2-stage vs single-stage: resolve total steps ─────────────────────
    # When --stage1_steps is set, use 2-stage adaptation (LIBERO-style):
    #   Stage 1: soft_prompt + action_out_proj only  (stage1_steps iterations)
    #   Stage 2: + Expert Gemma + action_in_proj     (stage2_steps iterations)
    # Otherwise fall back to the single-stage --finetune_steps.
    _use_2stage   = args.stage1_steps is not None
    _stage1_steps = args.stage1_steps if _use_2stage else 0
    _stage2_steps = args.stage2_steps if _use_2stage else 0
    _stage2_lr    = args.stage2_lr if args.stage2_lr is not None else args.lr
    total_finetune_steps = (_stage1_steps + _stage2_steps) if _use_2stage else args.finetune_steps

    if _is_main:
        if _use_2stage:
            log.info(
                "2-stage adaptation: stage1=%d steps (frozen backbone), "
                "stage2=%d steps (joint, lr=%.2e)  total=%d",
                _stage1_steps, _stage2_steps, _stage2_lr, total_finetune_steps,
            )
        else:
            log.info("Single-stage fine-tuning: %d steps", total_finetune_steps)

    accum_steps = args.gradient_accumulation_steps
    micro_step  = 0          # counts micro-batches within current accumulation cycle
    accum_loss  = 0.0        # running sum of (loss / accum_steps) for current cycle

    loader_iter = iter(loader)
    t0 = time.time()
    t_interval = time.time()  # wall-clock at start of current log interval

    if _is_main:
        log.info(
            "Starting Phase-2 fine-tuning for %d steps "
            "(gradient_accumulation_steps=%d, effective_batch=%d)",
            total_finetune_steps,
            accum_steps,
            args.batch_size * accum_steps,
        )

    # Zero gradients before the very first accumulation cycle
    optimizer.zero_grad(set_to_none=True)

    pbar = (
        tqdm.tqdm(
            total=total_finetune_steps,
            initial=start_step,
            desc="Finetune",
            dynamic_ncols=True,
        )
        if _is_main else None
    )

    while step < start_step + total_finetune_steps:
        # ── Infinite loader cycling ────────────────────────────────────────
        try:
            batch = next(loader_iter)
        except StopIteration:
            _loader_epoch += 1
            if use_ddp and sampler is not None:
                sampler.set_epoch(_loader_epoch)  # reshuffle differently each epoch
            loader_iter = iter(loader)
            batch = next(loader_iter)

        # ── Stage transition: Stage 1 → Stage 2 (2-stage mode only) ─────────
        rel_step = step - start_step   # 0-based step within this fine-tune run
        if (
            _use_2stage
            and not expert_unfrozen
            and rel_step >= _stage1_steps
        ):
            if _is_main:
                log.info(
                    "Step %d: Stage 2 start — unfreezing Expert Gemma + action_in_proj, "
                    "resetting LR to %.2e",
                    step, _stage2_lr,
                )
            setup_phase2_params(model_raw, unfreeze_expert=True)
            # Rebuild optimizer so newly unfrozen params are included
            trainable_params = [p for p in model_raw.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=_stage2_lr, weight_decay=1e-4)
            optimizer.zero_grad(set_to_none=True)
            expert_unfrozen = True
            if _is_main and args.wandb_enabled:
                _wandb_log({"event/expert_unfrozen": 1, "event/stage2_start": step}, step=step)

        # ── Legacy single-stage expert unfreeze (backward compat) ─────────
        elif (
            not _use_2stage
            and not expert_unfrozen
            and args.unfreeze_expert_step is not None
            and rel_step >= args.unfreeze_expert_step
        ):
            if _is_main:
                log.info("Step %d: unfreezing Expert Gemma + action_in_proj", step)
            setup_phase2_params(model_raw, unfreeze_expert=True)
            trainable_params = [p for p in model_raw.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
            optimizer.zero_grad(set_to_none=True)
            expert_unfrozen = True
            if _is_main and args.wandb_enabled:
                _wandb_log({"event/expert_unfrozen": 1}, step=step)

        # ── LR schedule (based on optimizer step) ─────────────────────────
        if _use_2stage:
            if rel_step < _stage1_steps:
                # Stage 1: cosine over stage1_steps
                current_lr = cosine_lr(
                    step=rel_step,
                    warmup=args.warmup_steps,
                    total=_stage1_steps,
                    peak=args.lr,
                )
            else:
                # Stage 2: cosine over stage2_steps, relative to stage transition
                rel_step_s2 = rel_step - _stage1_steps
                current_lr = cosine_lr(
                    step=rel_step_s2,
                    warmup=args.warmup_steps,
                    total=max(_stage2_steps, 1),
                    peak=_stage2_lr,
                )
        else:
            current_lr = cosine_lr(
                step=rel_step,
                warmup=args.warmup_steps,
                total=total_finetune_steps,
                peak=args.lr,
            )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # ── Forward + backward (with optional DDP no_sync) ────────────────
        obs, actions, domain_ids = batch_to_observation(batch, device)

        is_last_micro = (micro_step % accum_steps) == (accum_steps - 1)
        # Suppress DDP gradient sync on all but the last micro-step
        sync_ctx = (
            contextlib.nullcontext()
            if (not use_ddp or is_last_micro)
            else model.no_sync()
        )
        with sync_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                mse = model(phase=2, observation=obs, actions=actions, domain_ids=domain_ids)
            loss = mse.mean() / accum_steps   # scale so gradients are averaged
            loss.backward()

        accum_loss += loss.item()
        micro_step += 1

        if not is_last_micro:
            # Not yet time to step the optimizer — continue accumulating
            continue

        # ── Optimizer step (every accum_steps micro-batches) ───────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Average loss across ranks for accurate logging
        if use_ddp:
            _loss_t = torch.tensor(accum_loss, device=device)
            # Gloo backend does not support ReduceOp.AVG; use SUM/world_size so
            # loss averaging works for both NCCL and Gloo.
            dist.all_reduce(_loss_t, op=dist.ReduceOp.SUM)
            _loss_t /= dist.get_world_size()
            loss_item = _loss_t.item()
        else:
            loss_item = accum_loss
        loss_acc    += loss_item
        loss_sq_acc += loss_item ** 2
        accum_loss   = 0.0
        step += 1

        # ── Logging (main process only) ────────────────────────────────────
        if _is_main and step % args.log_interval == 0:
            n            = args.log_interval
            avg_loss     = loss_acc / n
            loss_std     = (max(loss_sq_acc / n - avg_loss ** 2, 0.0)) ** 0.5

            elapsed      = time.time() - t0
            steps_so_far = step - start_step
            steps_per_s  = steps_so_far / elapsed if elapsed > 0 else 0
            samples_per_s = steps_per_s * args.batch_size * accum_steps
            eta_s        = (total_finetune_steps - steps_so_far) / steps_per_s if steps_per_s > 0 else 0

            # interval-level timing (more accurate than cumulative avg)
            now = time.time()
            interval_steps_per_s = n / max(now - t_interval, 1e-6)
            t_interval = now

            epoch = (step * args.batch_size) / max(len(dataset), 1)

            log.info(
                "step %6d/%d | loss %.4f±%.4f | grad_norm %.3f | lr %.2e | "
                "%.1f steps/s | %.0f samp/s | ETA %.0fs",
                step,
                start_step + total_finetune_steps,
                avg_loss,
                loss_std,
                grad_norm.item(),
                current_lr,
                interval_steps_per_s,
                interval_steps_per_s * args.batch_size,
                eta_s,
            )
            if args.wandb_enabled:
                metrics = {
                    "train/loss":         avg_loss,
                    "train/loss_std":     loss_std,
                    "train/grad_norm":    grad_norm.item(),
                    "train/lr":           current_lr,
                    "train/epoch":        epoch,
                    "perf/steps_per_s":   interval_steps_per_s,
                    "perf/samples_per_s": interval_steps_per_s * args.batch_size * accum_steps
                                          * (dist.get_world_size() if use_ddp else 1),
                    "perf/eta_s":         eta_s,
                }
                metrics.update(_gpu_mem_stats())
                _wandb_log(metrics, step=step)
            if _metrics_file is not None:
                _row = {
                    "step":        step,
                    "loss":        round(avg_loss, 6),
                    "loss_std":    round(loss_std, 6),
                    "grad_norm":   round(float(grad_norm.item()), 6),
                    "lr":          current_lr,
                    "epoch":       round(epoch, 4),
                    "steps_per_s": round(interval_steps_per_s, 3),
                    "eta_s":       round(eta_s, 1),
                }
                _metrics_file.write(json.dumps(_row) + "\n")
            loss_acc    = 0.0
            loss_sq_acc = 0.0

        # ── Checkpoint (main process only) ────────────────────────────────
        if _is_main and step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, ckpt_dir, norm_stats=norm_stats, blocking=False)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_item:.4f}", lr=f"{current_lr:.2e}")

    if pbar is not None:
        pbar.close()

    # ── Final checkpoint (main process only) ──────────────────────────────
    if _is_main:
        log.info("Training done.  Saving final checkpoint at step %d …", step)
        save_checkpoint(model, optimizer, step, ckpt_dir, norm_stats=norm_stats, blocking=True)

        if _metrics_file is not None:
            _metrics_file.close()

        if args.wandb_enabled:
            _wandb_finish()

        log.info("Done. Checkpoints at: %s", ckpt_dir.resolve())
        log.info(
            "Evaluate with:\n"
            "  bash policy/Soft-VLA/eval.sh %s %s softvla_robotwin_ft %s %d <softvla_step> <seed> <gpu_id>\n"
            "  # <softvla_step> = number of actions to execute per chunk (e.g. %d = action_horizon)\n"
            "  # checkpoint_id = %d  (loaded via 5th argument above)",
            args.task_name,
            args.task_config,
            args.exp_name,
            step,
            args.action_horizon,
            step,
        )

    _cleanup_ddp()


if __name__ == "__main__":
    main()
