"""Per-dataset action normalization helpers for OXE pre-training."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np

ACTION_NORM_STATS_FILENAME = "oxe_action_norm_stats.json"
ACTION_NORM_STATS_VERSION = 2


def get_canonical_action_10d_normalization_mask() -> np.ndarray:
    """Return the default canonical 10D normalization mask.

    Canonical action layout is ``[xyz(3), r6(6), gripper(1)]``.
    Following OpenVLA's contract, motion dims are normalized while the
    absolute gripper dim is preserved as-is.
    """
    return np.asarray([True] * 9 + [False], dtype=bool)


def _resolve_normalization_mask(stats: dict[str, Any], action_dim: int) -> np.ndarray:
    mask_src = stats.get("normalization_mask")
    if mask_src is None:
        mask = np.ones(action_dim, dtype=bool)
    else:
        mask = np.asarray(mask_src, dtype=bool)
    if mask.shape != (action_dim,):
        raise ValueError(
            f"Normalization mask shape {mask.shape} does not match action dim ({action_dim},)"
        )
    return mask


def get_action_norm_stats_path(norm_stats_dir: str | None) -> str | None:
    if norm_stats_dir is None:
        return None
    return os.path.join(norm_stats_dir, ACTION_NORM_STATS_FILENAME)


def build_action_norm_entry(
    *,
    dataset_id: int,
    tfds_name: str,
    count: int,
    mean: np.ndarray,
    std: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    normalization_mask: np.ndarray | list[bool] | None = None,
) -> dict[str, Any]:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    mask = np.asarray(
        normalization_mask if normalization_mask is not None else get_canonical_action_10d_normalization_mask(),
        dtype=bool,
    )
    if mask.shape != mean.shape:
        raise ValueError(
            f"Normalization mask shape {mask.shape} does not match stats shape {mean.shape}"
        )
    return {
        "dataset_id": int(dataset_id),
        "tfds_name": tfds_name,
        "count": int(count),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "normalization_mask": mask.tolist(),
    }


def save_action_norm_stats(
    output_path: str,
    *,
    action_horizon: int,
    datasets: list[dict[str, Any]],
    norm_type: str = "quantile",
    normalization_mask: np.ndarray | list[bool] | None = None,
) -> None:
    mask = np.asarray(
        normalization_mask if normalization_mask is not None else get_canonical_action_10d_normalization_mask(),
        dtype=bool,
    )
    payload = {
        "version": ACTION_NORM_STATS_VERSION,
        "layout": "canonical_action_10d",
        "action_horizon": int(action_horizon),
        "norm_type": norm_type,
        "normalization_mask": mask.tolist(),
        "datasets": datasets,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


@lru_cache(maxsize=4)
def load_action_norm_stats(output_path: str) -> dict[str, Any]:
    with open(output_path) as f:
        payload = json.load(f)

    if payload.get("version") != ACTION_NORM_STATS_VERSION:
        raise ValueError(
            f"Unsupported action norm stats version: {payload.get('version')} "
            f"(expected {ACTION_NORM_STATS_VERSION})"
        )

    default_mask = None
    if payload.get("normalization_mask") is not None:
        default_mask = np.asarray(payload["normalization_mask"], dtype=bool)
        payload["normalization_mask"] = default_mask

    dataset_map: dict[int, dict[str, Any]] = {}
    for entry in payload.get("datasets", []):
        mean = np.asarray(entry["mean"], dtype=np.float32)
        std = np.asarray(entry["std"], dtype=np.float32)
        q01 = np.asarray(entry["q01"], dtype=np.float32)
        q99 = np.asarray(entry["q99"], dtype=np.float32)
        mask_src = entry.get("normalization_mask", default_mask)
        if mask_src is None:
            mask = np.ones(mean.shape[-1], dtype=bool)
        else:
            mask = np.asarray(mask_src, dtype=bool)
        if mask.shape != mean.shape:
            raise ValueError(
                f"Normalization mask shape {mask.shape} does not match stats shape {mean.shape}"
            )
        dataset_map[int(entry["dataset_id"])] = {
            "dataset_id": int(entry["dataset_id"]),
            "tfds_name": entry["tfds_name"],
            "count": int(entry["count"]),
            "mean": mean,
            "std": std,
            "q01": q01,
            "q99": q99,
            "normalization_mask": mask,
        }

    payload["dataset_map"] = dataset_map
    return payload


def normalize_action_array(
    actions_10d: np.ndarray,
    stats: dict[str, Any],
    *,
    norm_type: str,
    eps: float = 1e-6,
) -> np.ndarray:
    actions_10d = np.asarray(actions_10d, dtype=np.float32)
    out = actions_10d.astype(np.float32, copy=True)
    mask = _resolve_normalization_mask(stats, actions_10d.shape[-1])

    if norm_type == "quantile":
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        width = q99 - q01
        active = mask & (width > eps)
        if np.any(active):
            out[..., active] = (actions_10d[..., active] - q01[active]) / width[active] * 2.0 - 1.0
        zero_mask = mask & (width <= eps)
        if np.any(zero_mask):
            out[..., zero_mask] = 0.0
        return out
    if norm_type == "zscore":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        active = mask & (std > eps)
        if np.any(active):
            out[..., active] = (actions_10d[..., active] - mean[active]) / std[active]
        zero_mask = mask & (std <= eps)
        if np.any(zero_mask):
            out[..., zero_mask] = 0.0
        return out
    raise ValueError(f"Unknown action norm type: {norm_type}")


def normalize_action_batch(
    actions_10d: np.ndarray,
    dataset_ids: np.ndarray,
    stats_payload: dict[str, Any],
    *,
    norm_type: str,
) -> np.ndarray:
    dataset_ids = np.asarray(dataset_ids, dtype=np.int32).reshape(-1)
    if actions_10d.shape[0] != dataset_ids.shape[0]:
        raise ValueError(
            f"actions batch ({actions_10d.shape[0]}) and dataset_ids ({dataset_ids.shape[0]}) must align"
        )

    out = actions_10d.astype(np.float32, copy=True)
    dataset_map = stats_payload["dataset_map"]

    for dataset_id in np.unique(dataset_ids):
        dataset_id = int(dataset_id)
        if dataset_id not in dataset_map:
            raise KeyError(f"Dataset id {dataset_id} missing from action norm stats")
        mask = dataset_ids == dataset_id
        out[mask] = normalize_action_array(out[mask], dataset_map[dataset_id], norm_type=norm_type)
    return out