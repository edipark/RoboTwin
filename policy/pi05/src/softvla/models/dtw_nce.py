"""DTW-based soft InfoNCE loss for cross-robot latent alignment.

Paper formulation:
  1. Compute pairwise DTW distances on 10D action chunks (R6 rotation channels
     down-weighted by rotation_weight).
  2. CDF-normalize distances to [0, 1] using a precomputed empirical CDF
     (sorted distances from offline precompute step).
  3. Gaussian RBF kernel: s_ij = exp(-D̃² / 2σ²)  where D̃ ∈ [0, 1].
  4. Probability-normalize: w_ij = s_ij / Σ_k s_ik.
  5. Soft InfoNCE: L_i = -Σ_j w_ij · log_softmax(sim_ij / τ).

If no CDF is provided, raw DTW distances are used with the RBF kernel directly
(fallback mode — sigma should be tuned to the raw distance scale).
"""

import numpy as np
import torch
import torch.nn.functional as F

# ── Fast DTW backend (dtaidistance C extension) ───────────────────────────────
try:
    from dtaidistance import dtw_ndim as _dtai_ndim
    _dtai_ndim.distance_matrix_fast  # raises AttributeError if C ext not compiled
    _DTAI_AVAILABLE = True
except (ImportError, AttributeError):
    _DTAI_AVAILABLE = False


# ── DTW (on-the-fly, per-batch) ───────────────────────────────────────────────

def pairwise_dtw_distances_np(actions_np: np.ndarray) -> np.ndarray:
    """Compute pairwise DTW distance matrix for CPU numpy action trajectories.

    Args:
        actions_np: Action trajectories [B, H, D] as a CPU numpy array.

    Returns:
        Symmetric float32 distance matrix [B, B].
    """
    acts_np = np.ascontiguousarray(actions_np, dtype=np.float64)
    N, H, D = acts_np.shape
    rows, cols = np.triu_indices(N, k=1)

    if _DTAI_AVAILABLE:
        compact = _dtai_ndim.distance_matrix_fast(
            acts_np, ndim=D, compact=True, parallel=True
        )
        flat = np.asarray(compact, dtype=np.float32)
    else:
        # Vectorised numpy fallback: euclidean frame distances, sum over H
        # acts_np: [N, H, D] → pairwise L2 frame cost summed as DTW lower bound
        a = acts_np[rows]   # [P, H, D]
        b = acts_np[cols]   # [P, H, D]
        flat = np.sqrt(((a - b) ** 2).sum(axis=-1)).sum(axis=-1).astype(np.float32)

    dist_np = np.zeros((N, N), dtype=np.float32)
    dist_np[rows, cols] = flat
    dist_np[cols, rows] = flat
    return dist_np


def pairwise_dtw_distances(actions: torch.Tensor) -> torch.Tensor:
    """Compute pairwise DTW distance matrix for a batch of trajectories.

    Uses dtaidistance C+OpenMP extension when available (N=128 in <5ms),
    falling back to a pure-numpy vectorised path otherwise.
    Called inside torch.no_grad() so CPU round-trip is safe.

    Args:
        actions: Action trajectories [B, H, D]. May be on any device.

    Returns:
        Symmetric distance matrix [B, B] on the same device as *actions*.
    """
    device = actions.device
    acts_np = actions.detach().cpu().to(torch.float64).numpy()  # [B, H, D]
    dist_np = pairwise_dtw_distances_np(acts_np)
    return torch.from_numpy(dist_np).to(device=device)


# ── CDF normalization ────────────────────────────

def cdf_normalize(
    D: torch.Tensor,
    sorted_distances: torch.Tensor,
) -> torch.Tensor:
    """Map raw DTW distances to [0, 1] via empirical CDF.

    Args:
        D: Pairwise DTW distance matrix [B, B].
        sorted_distances: Pre-sorted 1D tensor of reference distances from
            offline precomputation.

    Returns:
        CDF-normalized distances [B, B] in [0, 1].
    """
    flat = D.flatten()
    # searchsorted on the sorted reference distances
    indices = torch.searchsorted(sorted_distances, flat)
    cdf_vals = indices.float() / len(sorted_distances)
    return cdf_vals.reshape(D.shape)


# ── Main loss function ───────────────────────────

def compute_dtw_nce(
    z: torch.Tensor,
    actions: torch.Tensor,
    tau: float = 0.1,
    sigma: float = 0.1,
    cdf_sorted_distances: torch.Tensor | None = None,
    rotation_weight: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft InfoNCE with DTW-distance-based weights (paper formulation).

    Args:
        z: Pooled latent embeddings [B, D_latent].
        actions: Ground-truth action trajectories [B, H, action_dim].
            First 10 dims used: xyz(3) + R6(6) + gripper(1).
        tau: Temperature for cosine similarity scaling.
        sigma: Gaussian RBF bandwidth (on CDF-normalized [0,1] distances).
        cdf_sorted_distances: Sorted 1D tensor of precomputed DTW distances
            for CDF normalization. If None, raw DTW distances are used.
        rotation_weight: Scale factor for R6 rotation channels (3:9) before DTW.
        eps: Numerical stability constant.

    Returns:
        loss: Scalar soft InfoNCE loss.
        stats: Dict with detached diagnostics.
    """
    device = z.device
    B = z.size(0)

    if B < 2:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"dtw_nce_loss": zero.detach(), "mean_dtw": torch.tensor(0.0)}

    # 1. Prepare action chunks: first 10D, apply rotation weight
    acts = actions[:, :, :10].clone().float()
    acts[:, :, 3:9] *= rotation_weight

    # 2. On-the-fly DTW distances
    with torch.no_grad():
        D = pairwise_dtw_distances(acts)

    mean_dtw_raw = D.sum() / (B * (B - 1))  # mean of off-diagonal

    # 3. CDF normalize if available
    if cdf_sorted_distances is not None:
        D = cdf_normalize(D, cdf_sorted_distances.to(device=device))

    # 4. Gaussian RBF kernel: s_ij = exp(-D̃² / 2σ²)
    eye = torch.eye(B, device=device, dtype=torch.bool)
    s = torch.exp(-D ** 2 / (2 * sigma ** 2))
    s = s.masked_fill(eye, 0.0)  # zero self-similarity

    # 5. Probability normalize: w_ij = s_ij / Σ_k s_ik
    w = s / (s.sum(dim=1, keepdim=True) + eps)

    # 6. Numerically stable soft InfoNCE
    emb = F.normalize(z, dim=-1)
    sim = emb @ emb.t() / tau  # [B, B]

    # Stability: subtract max of off-diagonal for each row
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=sim.dtype)
    sim_for_max = sim.masked_fill(eye, neg_inf)
    sim_stable = sim - sim_for_max.max(dim=1, keepdim=True).values

    # Log-softmax over off-diagonal entries
    log_prob = sim_stable - torch.logsumexp(
        sim_stable.masked_fill(eye, neg_inf), dim=1, keepdim=True
    )

    # Weighted sum: L_i = -Σ_j w_ij · log P(j|i)
    valid = w.sum(dim=1) > eps
    if valid.any():
        loss = -(w[valid] * log_prob[valid]).sum(dim=1).mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    stats = {
        "dtw_nce_loss": loss.detach(),
        "mean_dtw": mean_dtw_raw.detach(),
        "mean_weight": w.sum(dim=1).mean().detach(),
    }
    return loss, stats
