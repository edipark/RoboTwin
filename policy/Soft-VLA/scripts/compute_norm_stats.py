"""Compute and save normalization statistics from processed RoboTwin episodes.

Usage (from policy/Soft-VLA/):
    python scripts/compute_norm_stats.py \\
        --processed_dir processed_data/beat_block_hammer-demo_clean-50 \\
        --output_dir    assets/robotwin/beat_block_hammer-demo_clean-50

The output directory will contain norm_stats.json (mean, std, q01, q99 for
"state" and "actions" keys).  The finetune.py script runs this automatically
when --norm_stats_dir is not specified.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

# ── Resolve Soft-VLA src ──────────────────────────────────────────────────────
_POLICY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_POLICY_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from openpi.shared import normalize as _normalize  # noqa: E402


def compute_and_save(processed_dir: str, output_dir: str, action_horizon: int = 16) -> dict:
    """Scan all episodes in *processed_dir*, compute RunningStats, and save.

    Returns:
        dict[str, NormStats] with keys "state" and "actions" (plural — matches the
        upstream OpenPI convention that ``Policy.infer`` puts in its output dict).
    """
    state_stats  = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()

    ep_dirs = sorted(
        [
            d
            for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
            and d.startswith("episode_")
        ],
        key=lambda x: int(x.split("_")[1]),
    )

    if not ep_dirs:
        raise ValueError(f"No episode_ directories found in {processed_dir}")

    for ep_dir in ep_dirs:
        hdf5_path = os.path.join(processed_dir, ep_dir, f"{ep_dir}.hdf5")
        if not os.path.isfile(hdf5_path):
            print(f"  [warn] missing {hdf5_path}, skipping")
            continue
        with h5py.File(hdf5_path, "r") as f:
            qpos    = f["observations/qpos"][()].astype(np.float32)  # [T, D] — absolute EE state
            actions = f["action"][()].astype(np.float32)             # [T, D] — absolute EE next-state

        # Actions on disk are absolute xyz (stored by process_data.py).
        # RoboTwinDataset applies CHUNK-WISE delta at load time:
        #   action_chunk[:, :3] -= state[:3]   (state_t subtracted from ALL horizon steps)
        # So for each timestep t, the model sees:
        #   action_chunk[k, :3] = actions[t+k, :3] - qpos[t, :3]  (cumulative delta from state_t)
        # rot6d (dims 3-8) and gripper (dim 9) remain absolute.
        # We iterate over all valid (t, k) pairs to match the true distribution.
        T = qpos.shape[0]
        for t in range(T):
            end = min(t + action_horizon, T)
            chunk = actions[t:end].copy()                    # [chunk_len, D]
            chunk[:, :3] -= qpos[t, :3]                     # chunk-wise delta
            # temporal padding: repeat last row
            if chunk.shape[0] < action_horizon:
                pad = np.tile(chunk[-1:], (action_horizon - chunk.shape[0], 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            action_stats.update(chunk)                       # [action_horizon, D]

        state_stats.update(qpos)

    norm_stats = {
        "state":   state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    os.makedirs(output_dir, exist_ok=True)
    _normalize.save(output_dir, norm_stats)

    print(f"Saved norm stats → {os.path.abspath(output_dir)}")
    print(f"  state   mean[:6]: {norm_stats['state'].mean[:6]}")
    print(f"  state   std[:6]:  {norm_stats['state'].std[:6]}")
    print(f"  actions mean[:6]: {norm_stats['actions'].mean[:6]}")
    print(f"  actions std[:6]:  {norm_stats['actions'].std[:6]}")

    return norm_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for Soft-VLA fine-tuning."
    )
    parser.add_argument(
        "--processed_dir",
        required=True,
        help="Path to processed_data/<task>-<config>-<N>/",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where norm_stats.json will be written",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=16,
        help="Action chunk length (must match model, default 16).",
    )
    args = parser.parse_args()
    compute_and_save(args.processed_dir, args.output_dir, action_horizon=args.action_horizon)
