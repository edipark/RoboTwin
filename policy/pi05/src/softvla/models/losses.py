"""Phase 1 loss functions for Soft-VLA base.

L_soft_nce: DTW-guided soft InfoNCE contrastive alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase1Loss(nn.Module):
    """Container for Phase 1 loss functions.

    Stateless (no learned parameters); the class exists purely
    for organisational convenience and to allow easy subclassing.
    """

    # ── Soft InfoNCE loss ─────────────────────────

    @staticmethod
    def l_soft_nce(
        z_student: torch.Tensor,
        dtw_weights: torch.Tensor,
        tau: float = 0.1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """DTW-guided soft InfoNCE loss.

        Uses pre-computed, row-normalised DTW similarity weights as soft
        labels.  Encourages embeddings of behaviourally similar trajectories
        to be close while pushing apart those that are behaviourally dissimilar.

        Args:
            z_student: Student latent embeddings [B, D].
            dtw_weights: Row-normalised similarity matrix [B, B].
                         Diagonal should be 0.  Caller is responsible for
                         computing this from raw DTW distances via RBF kernel
                         and row-normalisation.
            tau: Contrastive temperature.
            eps: Numerical stability constant (not used directly here because
                 we rely on the LogSumExp trick via log_softmax).

        Returns:
            Scalar soft InfoNCE loss.
        """
        B = z_student.size(0)
        if B < 2:
            return z_student.new_zeros(())

        # Normalised cosine similarity matrix [B, B] scaled by temperature
        emb = F.normalize(z_student, dim=-1)
        sim = emb @ emb.t() / tau  # [B, B]

        # Numerically stable log softmax per row (LogSumExp trick via F.log_softmax)
        # Mask the diagonal to exclude self from the denominator as per standard NCE.
        eye = torch.eye(B, device=sim.device, dtype=torch.bool)
        sim_for_softmax = sim.masked_fill(eye, float("-inf"))
        log_softmax_sim = F.log_softmax(sim_for_softmax, dim=1)  # [B, B]

        # Soft cross-entropy: L_i = -Σ_j w_ij * log p(j | i)
        # dtw_weights already row-normalised with diagonal=0.
        # NUMERICAL SAFETY: dtw_weights[i,i]=0 but log_softmax_sim[i,i]=-inf,
        # so the product is 0*(-inf)=NaN.  Mask log values where weight==0
        # before multiplying so those entries contribute exactly 0.
        log_softmax_safe = log_softmax_sim.masked_fill(dtw_weights == 0, 0.0)

        # Per-row weighted cross-entropy.
        # denom = row sum of dtw_weights.  For rows that are already row-normalised,
        # denom ≈ 1.0.  For rows where ALL weights are zero (no positive pairs in the
        # batch), denom = 0 — these rows must be excluded from the mean.
        # Including zero-positive rows pushes toward uniform log-softmax (maximise entropy),
        # which directly conflicts with the contrastive objective.  This is the CLASS fix
        # (base_policy.py: valid_samples_mask = denom > 1e-6).
        per_row = -(dtw_weights * log_softmax_safe).sum(dim=1)  # [B]
        denom = dtw_weights.sum(dim=1)                          # [B]  ≈ 1.0 for valid rows
        valid = denom > eps                                     # rows with at least one positive
        if not valid.any():
            # No positive pairs in this batch — return a zero loss that still
            # carries a grad_fn so backward() does not raise.
            # Multiplying by z_student keeps the computation graph alive without
            # contributing any actual gradient signal.
            return z_student.mean() * 0.0
        loss = (per_row[valid] / denom[valid]).mean()
        return loss

    @staticmethod
    @torch.no_grad()
    def compute_alignment_uniformity(
        z_student: torch.Tensor,
        dtw_weights: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute alignment and uniformity metrics for contrastive learning monitoring.

        Alignment : mean cosine similarity of *positive* pairs (dtw_weights > 0).
                    Should rise from ~0.3-0.4 toward 0.7-0.9 during training.
        Uniformity: mean cosine similarity of *negative* pairs (dtw_weights == 0,
                    off-diagonal).  Should fall toward 0 or below during training.

        Args:
            z_student:   Raw (un-normalised) embeddings [B, D].
            dtw_weights: Row-normalised similarity matrix [B, B] with zero diagonal.

        Returns:
            (alignment, uniformity) as Python floats.
        """
        B = z_student.size(0)
        if B < 2:
            return 0.0, 0.0

        # float32 for numerical stability (bfloat16 cosine values are coarse)
        emb = F.normalize(z_student.detach().float(), dim=-1)  # [B, D]
        sim = emb @ emb.t()                                     # [B, B]

        eye = torch.eye(B, device=sim.device, dtype=torch.bool)
        off_diag = ~eye

        pos_mask = (dtw_weights > 0) & off_diag
        neg_mask = (dtw_weights == 0) & off_diag

        alignment  = sim[pos_mask].mean().item() if pos_mask.any()  else 0.0
        uniformity = sim[neg_mask].mean().item() if neg_mask.any() else 0.0
        return alignment, uniformity

    @staticmethod
    @torch.no_grad()
    def compute_target_entropy(
        dtw_weights: torch.Tensor,
        eps: float = 1e-8,
    ) -> float:
        """Shannon entropy of the DTW soft-label distribution, averaged over valid anchors.

        This is the theoretical lower bound of l_soft_nce.  When
        loss_nce ≈ target_entropy, D_KL(p||q) ≈ 0, meaning the model
        has fully learned the DTW gravity field.

        Args:
            dtw_weights: Row-normalised similarity matrix [B, B] with zero diagonal.
            eps: Small constant for numerical stability in log.

        Returns:
            Mean H(p) over valid rows as a Python float.
        """
        denom = dtw_weights.sum(dim=1)
        valid = denom > eps
        if not valid.any():
            return 0.0
        p = dtw_weights[valid]  # already row-normalised; diagonal is 0
        entropy_per_anchor = -(p * torch.log(p + eps)).sum(dim=1)
        return float(entropy_per_anchor.mean().item())

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use l_soft_nce() directly.")


# ── DTW weight computation (called outside autograd) ──────────────────────────

def compute_dtw_weights(
    actions: torch.Tensor,
    rotation_weight: float = 0.5,
    gripper_weight: float = 1.0,
    translation_only: bool = False,
    max_cdf: float = 1.0,
    cdf_sorted_distances: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute row-normalised DTW similarity weights for a batch of trajectories.

    This function is intended to be called inside ``torch.no_grad()`` before
    the differentiable forward pass.

    Action channel layout (first 10 dims):
        [0:3]  xyz translation
        [3:9]  R6 rotation
        [9]    gripper

    Args:
        actions: Ground-truth action trajectories [B, H, action_dim].
        rotation_weight: Scale factor for R6 channels (indices 3:9). Ignored when
            ``translation_only=True``.
        gripper_weight: Scale factor for the gripper channel (index 9). Set to 0
            to exclude gripper entirely. Ignored when ``translation_only=True``.
        translation_only: If True, use only the xyz translation channels (0:3) for
            DTW, ignoring rotation and gripper.
        max_cdf: CDF cutoff in (0, 1].  Pairs whose CDF-normalised distance exceeds
            this threshold receive zero weight (pure negatives).
        cdf_sorted_distances: Pre-sorted 1-D tensor of reference DTW distances
            for empirical CDF normalisation.  If None, batch-local normalisation
            is used (suitable for fake_data / unit-tests).
        eps: Small constant for safe row normalisation.

    Returns:
        Row-normalised weight matrix [B, B] with zero diagonal.
    """
    from softvla.models.dtw_nce import pairwise_dtw_distances, cdf_normalize  # local import to avoid cycles

    B = actions.size(0)
    device = actions.device

    if B < 2:
        return torch.zeros(B, B, device=device)

    # Prepare action features with per-channel weighting
    acts = actions[:, :, :10].clone().float()
    if translation_only:
        # Use only xyz (channels 0:3); zero out rotation and gripper
        acts[:, :, 3:] = 0.0
    else:
        acts[:, :, 3:9] *= rotation_weight   # R6 rotation
        acts[:, :, 9:10] *= gripper_weight   # gripper (slice to keep dim)

    # Pairwise DTW [B, B]
    D = pairwise_dtw_distances(acts)

    # Optional CDF normalisation to [0, 1]
    if cdf_sorted_distances is not None:
        D = cdf_normalize(D, cdf_sorted_distances.to(device=device))
    else:
        # No global reference: normalise within the batch so that D ∈ [0, 1].
        # This mirrors CLASS's global get_cdf_dist but at batch scope.
        # Necessary for fake_data / unit-tests where no dtw_cdf_path is provided.
        triu = torch.triu(D, diagonal=1)
        d_max = triu[triu > 0].max() if (triu > 0).any() else D.new_tensor(1.0)
        D = D / (d_max + eps)

    # ── CLASS-style linear CDF weight (replaces Gaussian RBF) ──────────────
    # CLASS (get_cdf_dist): w_ij = 1 - rank/k for the top-k% closest pairs,
    # 0 for the rest.  This is linear and scale-invariant, so it doesn't
    # require tuning sigma. Gaussian RBF was extremely sensitive to sigma:
    # with sigma=0.04 and CDF-normalised D∈[0,1], exp(-D²/0.0032)≈0 for D>0.1.
    #
    # Implementation:
    #   1. Threshold at max_cdf (equivalent to CLASS dist_quantile).
    #   2. Within threshold, assign w = (max_cdf - D) / max_cdf → linear [0,1].
    #      Closest pair gets w≈1, pairs at max_cdf boundary get w≈0.
    #   3. Zero the diagonal and row-normalise.
    eye = torch.eye(B, device=device, dtype=torch.bool)
    D_diag_inf = D.masked_fill(eye, float("inf"))

    # Pairs beyond max_cdf become pure negatives (weight=0)
    beyond = D_diag_inf > max_cdf
    # Linear similarity: 1 at D=0, 0 at D=max_cdf
    s = ((max_cdf - D_diag_inf) / (max_cdf + eps)).clamp(min=0.0)
    s = s.masked_fill(beyond, 0.0)

    # Row-normalise (rows with all-zero weights stay zero — no positives in batch)
    w = s / (s.sum(dim=1, keepdim=True) + eps)
    return w
