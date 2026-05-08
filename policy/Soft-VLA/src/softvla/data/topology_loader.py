"""Topology-Aware OXE Data Loader.

Assembles batches so that each *group* of ``group_size`` trajectories contains
one anchor and (group_size - 1) nearest neighbours (by DTW distance on actions).
This forces the DTW-NCE loss to work with hard positive pairs instead of
relying on random chance.

Algorithm
---------
1. Draw ``buffer_size`` steps from the base loader into a ring buffer.
2. Compute the pairwise DTW distance matrix [N, N] via dtaidistance (C+OpenMP).
   N=2048 takes ~0.5s; fully hidden behind the async prefetch queue.
3. Assemble batches: divide N samples into groups of ``group_size``, each with
   one anchor + (group_size-1) nearest neighbours.
4. Yield batches in the same format as the base loader.
5. When the buffer is exhausted, refill transparently.

``batch_size`` must be divisible by ``group_size``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from softvla.data.rotation_utils import action_7d_to_10d

# ── Fast DTW backend ──────────────────────────────────────────────────────────
# Primary: dtaidistance (C extension, numpy 1.x compatible, multi-thread)
# Fallback: vectorised numpy (no external dependency, single process)
try:
    from dtaidistance import dtw_ndim as _dtai_ndim
    # Verify C extension is compiled (distance_matrix_fast raises if not)
    _dtai_ndim.distance_matrix_fast  # AttributeError → no C ext
    _DTAI_AVAILABLE = True
except (ImportError, AttributeError):
    _DTAI_AVAILABLE = False

_FALLBACK_WARNED = False

# torchrun forcibly sets OMP_NUM_THREADS=1 to prevent CPU oversubscription,
# but dtaidistance (OpenMP DTW) benefits from more threads since it runs in
# a background daemon thread and doesn't compete with PyTorch's CPU ops.
# Override to 8: each of the 8 rank processes gets 8 DTW threads → 64 total
# across all ranks, safely within a 72-core B200 node.
import os as _os_omp
_OMP_DTW_THREADS = int(_os_omp.environ.get("OMP_DTW_THREADS", "8"))
del _os_omp


# ── Pairwise DTW (fast path + numpy fallback) ──────────────────────────────────

def _vectorised_dtw_flat(acts: np.ndarray, chunk_size: int = 50_000) -> np.ndarray:
    """Vectorised numpy DTW for all N*(N-1)/2 upper-triangle pairs.

    Fallback when dtaidistance is not installed.  O(N²·H²) time but fully
    vectorised — much faster than a per-pair Python loop.

    Args:
        acts: [N, H, D] float32 (already feature-weighted).

    Returns:
        1-D float32 array of upper-triangle distances in row-major order.
    """
    N, H, _ = acts.shape
    rows_idx, cols_idx = np.triu_indices(N, k=1)
    P = len(rows_idx)
    distances = np.empty(P, dtype=np.float32)
    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        r_c = rows_idx[start:end]
        c_c = cols_idx[start:end]
        a = acts[r_c]  # [C, H, D]
        b = acts[c_c]
        diff = a[:, :, np.newaxis, :] - b[:, np.newaxis, :, :]  # [C, H, H, D]
        cost = np.sqrt((diff ** 2).sum(axis=-1))                 # [C, H, H]
        C = end - start
        INF = np.float32(np.inf)
        dtw = np.full((C, H + 1, H + 1), INF, dtype=np.float32)
        dtw[:, 0, 0] = 0.0
        for i in range(1, H + 1):
            for j in range(1, H + 1):
                prev = np.minimum(
                    np.minimum(dtw[:, i - 1, j], dtw[:, i, j - 1]),
                    dtw[:, i - 1, j - 1],
                )
                dtw[:, i, j] = cost[:, i - 1, j - 1] + prev
        distances[start:end] = dtw[:, H, H]
    return distances


def _fast_pairwise_dtw(acts: np.ndarray) -> np.ndarray:
    """Compute symmetric [N, N] DTW distance matrix.

    Uses dtaidistance (C-extension + OpenMP) when available for ~50-100×
    speedup over the vectorised numpy path.  Falls back to vectorised numpy.

    dtaidistance N-D format expects an iterable of per-sample [H, D] series.
    Passing a contiguous [N, H, D] float64 array preserves the fast
    multidimensional C/OpenMP path. ``compact=True`` returns the upper-triangle
    values in the same row-major order as ``np.triu_indices(N, k=1)``.

    Args:
        acts: [N, H, D] float32 (already feature-weighted).

    Returns:
        Symmetric [N, N] float32 distance matrix with zero diagonal.
    """
    N, H, D = acts.shape
    t0 = time.perf_counter()

    rows_idx, cols_idx = np.triu_indices(N, k=1)

    global _FALLBACK_WARNED

    if _DTAI_AVAILABLE:
        acts_ndim = np.ascontiguousarray(acts, dtype=np.double)
        # Temporarily raise OMP_NUM_THREADS for this DTW call.
        # torchrun sets it to 1 globally; we restore it afterward so PyTorch
        # DataLoader workers are unaffected.
        import os as _os_t
        _prev_omp = _os_t.environ.get("OMP_NUM_THREADS")
        _os_t.environ["OMP_NUM_THREADS"] = str(_OMP_DTW_THREADS)
        try:
            compact = _dtai_ndim.distance_matrix_fast(
                acts_ndim, ndim=D, compact=True, parallel=True
            )
        finally:
            if _prev_omp is None:
                _os_t.environ.pop("OMP_NUM_THREADS", None)
            else:
                _os_t.environ["OMP_NUM_THREADS"] = _prev_omp
        flat = np.asarray(compact, dtype=np.float32)
        backend = f"dtaidistance(C+parallel,omp={_OMP_DTW_THREADS})"
    else:
        if not _FALLBACK_WARNED:
            logging.warning(
                "[Topology] dtaidistance C/OpenMP backend unavailable; falling back to numpy DTW. "
                "Install dtaidistance to restore fast topology refill performance."
            )
            _FALLBACK_WARNED = True
        flat = _vectorised_dtw_flat(acts)
        backend = "numpy-vectorised"

    D_mat = np.zeros((N, N), dtype=np.float32)
    D_mat[rows_idx, cols_idx] = flat
    D_mat[cols_idx, rows_idx] = flat

    elapsed = time.perf_counter() - t0
    logging.debug(
        "[Topology] DTW %d×%d (%d pairs) in %.1fs [%s]", N, N, len(flat), elapsed, backend
    )
    return D_mat


class TopologyAwareOXELoader:
    """Wraps a base OXEDataLoader and re-assembles topology-aware batches.

    Args:
        base_loader:        An iterable that yields
                            ``(obs_dict, actions [B,H,D], domain_ids [B])``.
        buffer_size:        Number of *individual* samples to buffer.
                            Must be >= 2 * batch_size and ideally a multiple
                            of batch_size for clean refills.
        batch_size:         Number of samples per output batch.
        group_size:         Anchor + neighbours per group
                            (batch_size must be divisible by group_size).
        rotation_weight:    Down-weight for R6 channels (3:9) in DTW.
        gripper_weight:     Scale for gripper channel (9) in DTW.
        translation_only:   If True, only use xyz channels (0:3) for DTW.
        prefetch_batches:   Number of assembled batches to queue in the
                            background prefetch thread.  Default 8.
        cross_domain_only:  If True, nearest neighbours are forced to come
                            from *different* robot domains than the anchor.
                            Falls back to any-domain if no cross-domain
                            neighbours exist in the buffer.  Default False.
    """

    def __init__(
        self,
        base_loader: Any,
        buffer_size: int,
        batch_size: int,
        group_size: int,
        rotation_weight: float = 0.1,
        gripper_weight: float = 0.1,
        translation_only: bool = True,
        prefetch_batches: int = 8,
        cross_domain_only: bool = False,
    ):
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by group_size ({group_size})"
            )
        if buffer_size < 2 * batch_size:
            raise ValueError(
                f"buffer_size ({buffer_size}) must be >= 2 * batch_size ({2 * batch_size})"
            )

        self._base_iter = iter(base_loader)
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._group_size = group_size
        self._rotation_weight = rotation_weight
        self._gripper_weight = gripper_weight
        self._translation_only = translation_only
        self._prefetch_topology_batches = prefetch_batches
        self._cross_domain_only = cross_domain_only

        # Ring buffer — stored as lists of per-sample dicts
        self._buf_obs: list[dict] = []      # per-sample obs dicts
        self._buf_actions: list[np.ndarray] = []  # [H, D] each
        self._buf_domain_ids: list[int] = []
        self._buf_indices: list[int] = []   # permuted indices ready to emit
        self._buf_dtw: np.ndarray | None = None  # [N, N] precomputed distances
        self._buf_cross_domain: np.ndarray | None = None  # [N, N] bool cross-domain mask

        # Cross-domain fallback counter (read from training loop for logging).
        # Incremented each time _assemble_batch falls back to any-domain because
        # no cross-domain neighbours were available for an anchor.
        self.xdomain_fallback_count: int = 0

    # ── Buffer management ─────────────────────────────────────────────────

    def _draw_from_base(self, n_samples: int) -> list[tuple[dict, np.ndarray, int]]:
        """Pull *n_samples* individual frames from the base loader.

        Images are stored as uint8 in the buffer to reduce memory footprint
        ~4× (3.5 GB float32 → 0.9 GB uint8 for buffer_size=2048).  The
        float32 normalisation ( / 127.5 - 1.0 ) is deferred to _collate().
        """
        collected: list[tuple[dict, np.ndarray, int]] = []
        _log_interval = max(1, n_samples // 4)
        while len(collected) < n_samples:
            if len(collected) > 0 and len(collected) % _log_interval == 0:
                logging.debug("[Topology] Drawing samples: %d / %d …", len(collected), n_samples)
            obs_dict, actions_batch, domain_batch = next(self._base_iter)
            B = actions_batch.shape[0]
            for i in range(B):
                # Re-quantise float32 [-1,1] → uint8 [0,255] for compact storage.
                # Round-trip error is ±1/255 ≈ 0.004, negligible for training.
                obs_i = {
                    "images": {
                        k: np.clip((v[i] + 1.0) * 127.5, 0, 255).astype(np.uint8)
                        for k, v in obs_dict["images"].items()
                    },
                    "image_masks": {k: v[i] for k, v in obs_dict["image_masks"].items()},
                    "state": obs_dict["state"][i],
                    "tokenized_prompt": obs_dict["tokenized_prompt"][i],
                    "tokenized_prompt_mask": obs_dict["tokenized_prompt_mask"][i],
                }
                collected.append((obs_i, actions_batch[i], int(domain_batch[i])))
                if len(collected) >= n_samples:
                    break
        return collected

    def _refill_buffer(self) -> None:
        """Fill the buffer with buffer_size fresh samples and compute DTW."""
        # Stagger refill start across ranks to prevent concurrent NFS I/O storms
        # when all 8 DDP ranks refill at the same time.  A random 0-5s jitter
        # spreads the load so the filesystem is not hammered by 8 simultaneous
        # 4096-sample draws.
        import random as _random
        _jitter = _random.uniform(0.0, 5.0)
        if _jitter > 0.1:
            time.sleep(_jitter)

        _t0 = time.perf_counter()
        logging.debug("[Topology] Refilling buffer — drawing %d samples from base loader …", self._buffer_size)
        samples = self._draw_from_base(self._buffer_size)
        _t1 = time.perf_counter()

        self._buf_obs = [s[0] for s in samples]
        self._buf_actions = [s[1] for s in samples]
        self._buf_domain_ids = [s[2] for s in samples]

        # ── Compute pairwise DTW on 10D action chunks ─────────────────────
        # actions shape: [N, H, D] where D >= 7
        actions_np = np.stack(self._buf_actions, axis=0)  # [N, H, D]
        N = actions_np.shape[0]

        # Convert to 10D if original dim is 7; otherwise assume already 10+D
        if actions_np.shape[-1] == 7:
            actions_10d = action_7d_to_10d(actions_np.reshape(-1, actions_np.shape[-1]))
            actions_10d = actions_10d.reshape(N, actions_np.shape[1], 10)
        else:
            actions_10d = actions_np[..., :10]

        # Apply channel weighting before DTW
        acts = actions_10d.copy().astype(np.float32)
        if self._translation_only:
            acts = acts[..., :3]
        else:
            acts[..., 3:9] *= self._rotation_weight
            acts[..., 9:10] *= self._gripper_weight

        dtw_mat = _fast_pairwise_dtw(acts)  # [N, N]

        self._buf_dtw = dtw_mat
        self._buf_indices = list(np.random.permutation(N))

        # Precompute cross-domain mask [N, N] for nearest-neighbour filtering
        if self._cross_domain_only:
            doms = np.array(self._buf_domain_ids, dtype=np.int32)
            self._buf_cross_domain = (doms[:, None] != doms[None, :])
        else:
            self._buf_cross_domain = None

        _t2 = time.perf_counter()
        _draw_s = _t1 - _t0
        _dtw_s = _t2 - _t1
        _total_s = _t2 - _t0
        logging.info(
            "[Topology] Buffer refill: draw=%.1fs  dtw=%.1fs  total=%.1fs  (N=%d, %d batches)",
            _draw_s, _dtw_s, _total_s, N, len(self._buf_indices) // self._batch_size,
        )

    # ── Batch assembly ────────────────────────────────────────────────────

    def _assemble_batch(self) -> tuple[dict, np.ndarray, np.ndarray] | None:
        """Assemble one batch of batch_size samples from the buffer.

        Each group of group_size contains one anchor + (group_size-1) nearest
        neighbours from the rest of the buffer.

        Returns None when there are fewer than batch_size anchors left.
        """
        groups_per_batch = self._batch_size // self._group_size
        if len(self._buf_indices) < groups_per_batch:
            return None

        batch_indices: list[int] = []
        for _ in range(groups_per_batch):
            anchor_idx = self._buf_indices.pop(0)
            # Find (group_size-1) nearest neighbours excluding the anchor itself
            distances = self._buf_dtw[anchor_idx].copy()
            distances[anchor_idx] = np.inf
            # Also exclude already-selected samples to avoid duplicates
            for used in batch_indices:
                distances[used] = np.inf

            # Cross-domain: mask out same-domain candidates so positive pairs
            # always span different robot embodiments.
            if self._cross_domain_only and self._buf_cross_domain is not None:
                distances[~self._buf_cross_domain[anchor_idx]] = np.inf

            nn_count = min(self._group_size - 1, int(np.isfinite(distances).sum()))
            if nn_count == 0 and self._cross_domain_only:
                # No cross-domain neighbours available — fall back to any-domain
                self.xdomain_fallback_count += 1
                fallback = self._buf_dtw[anchor_idx].copy()
                fallback[anchor_idx] = np.inf
                for used in batch_indices:
                    fallback[used] = np.inf
                nn_count_fb = min(self._group_size - 1, int(np.isfinite(fallback).sum()))
                if nn_count_fb == 0:
                    nn_indices = [anchor_idx] * (self._group_size - 1)
                else:
                    nn_indices = list(np.argpartition(fallback, nn_count_fb)[:nn_count_fb])
            elif nn_count == 0:
                # Fallback: just replicate anchor
                nn_indices = [anchor_idx] * (self._group_size - 1)
            else:
                nn_indices = list(np.argpartition(distances, nn_count)[:nn_count])
            batch_indices.append(anchor_idx)
            batch_indices.extend(nn_indices)

        # Collate selected samples into batch arrays
        return self._collate(batch_indices)

    def get_xdomain_stats(self) -> dict[str, int]:
        """Return cross-domain statistics for logging (thread-safe read)."""
        count = self.xdomain_fallback_count
        self.xdomain_fallback_count = 0
        return {"topology_xdomain_fallback": count}

    def _collate(self, indices: list[int]) -> tuple[dict, np.ndarray, np.ndarray]:
        obs_list = [self._buf_obs[i] for i in indices]
        act_list = [self._buf_actions[i] for i in indices]
        dom_list = [self._buf_domain_ids[i] for i in indices]

        # Stack images; buffer stores uint8 — restore float32 [-1,1] here.
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        for cam_key in obs_list[0]["images"]:
            imgs_u8 = np.stack([o["images"][cam_key] for o in obs_list], axis=0)  # [B, H, W, 3] uint8
            images[cam_key] = imgs_u8.astype(np.float32) / 127.5 - 1.0
        for cam_key in obs_list[0]["image_masks"]:
            image_masks[cam_key] = np.stack([o["image_masks"][cam_key] for o in obs_list], axis=0)

        obs_dict = {
            "images": images,
            "image_masks": image_masks,
            "state": np.stack([o["state"] for o in obs_list], axis=0),
            "tokenized_prompt": np.stack([o["tokenized_prompt"] for o in obs_list], axis=0),
            "tokenized_prompt_mask": np.stack([o["tokenized_prompt_mask"] for o in obs_list], axis=0),
        }
        actions = np.stack(act_list, axis=0)
        domain_ids = np.array(dom_list, dtype=np.int32)
        return obs_dict, actions, domain_ids

    # ── Iterator ──────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[tuple[dict, np.ndarray, np.ndarray]]:
        """Yield topology-aware batches with async background prefetch.

        A background daemon thread fills the batch queue by running
        ``_refill_buffer`` (the slow O(N²) DTW step) + ``_assemble_batch``
        while the training loop consumes already-assembled batches.  This
        hides the DTW latency behind GPU computation so GPU utilisation
        stays high.

        The queue holds up to ``_prefetch_topology_batches`` assembled batches
        (default 8).  Sentinel ``None`` signals that the producer raised an
        exception (stored in ``_producer_exc``).
        """
        _PREFETCH = getattr(self, "_prefetch_topology_batches", 8)
        _SENTINEL = None
        _batch_q: queue.Queue = queue.Queue(maxsize=_PREFETCH)
        _exc_holder: list[BaseException | None] = [None]

        def _producer() -> None:
            try:
                while True:
                    if not self._buf_indices:
                        self._refill_buffer()

                    batch = self._assemble_batch()
                    if batch is None:
                        self._refill_buffer()
                        batch = self._assemble_batch()
                        if batch is None:
                            continue

                    _batch_q.put(batch)
            except Exception as exc:  # noqa: BLE001
                _exc_holder[0] = exc
                _batch_q.put(_SENTINEL)  # wake consumer

        t = threading.Thread(target=_producer, daemon=True, name="topology-prefetch")
        t.start()

        while True:
            item = _batch_q.get()
            if item is _SENTINEL:
                exc = _exc_holder[0]
                if exc is not None:
                    raise RuntimeError("Topology prefetch thread failed") from exc
                return  # producer signalled clean stop (shouldn't happen in infinite loop)
            yield item
