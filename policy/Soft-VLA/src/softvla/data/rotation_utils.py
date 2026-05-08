"""Rotation utilities for converting OXE 7D actions to 10D representation.

OXE datasets use 7D end-effector actions:
    [delta_x, delta_y, delta_z,  delta_rx, delta_ry, delta_rz,  gripper]
     xyz (3)                      Euler ZYX deltas (3)            (1)

dtw_nce.py expects first 10 dims laid out as:
    [x, y, z,  r0, r1, r2, r3, r4, r5,  gripper]
     xyz (3)   6D rotation (6)             (1)

The 6D rotation representation (Zhou et al., 2019) is the first two columns of
the rotation matrix R stored as a flat 6-vector: [R[:,0]; R[:,1]].
"""

from __future__ import annotations

import numpy as np


# ── Euler → Rotation matrix (ZYX / extrinsic XYZ convention) ───────────────

def euler_to_rotation_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert Euler angles (extrinsic XYZ = intrinsic ZYX) to rotation matrices.

    Args:
        euler_xyz: [..., 3] array of (roll, pitch, yaw) angles in radians.

    Returns:
        R: [..., 3, 3] rotation matrices.
    """
    rx, ry, rz = euler_xyz[..., 0], euler_xyz[..., 1], euler_xyz[..., 2]

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    # Rx
    ones = np.ones_like(rx)
    zeros = np.zeros_like(rx)
    Rx = np.stack([
        ones,  zeros,  zeros,
        zeros, cos_x, -sin_x,
        zeros, sin_x,  cos_x,
    ], axis=-1).reshape(euler_xyz.shape[:-1] + (3, 3))

    # Ry
    Ry = np.stack([
        cos_y,  zeros, sin_y,
        zeros,  ones,  zeros,
        -sin_y, zeros, cos_y,
    ], axis=-1).reshape(euler_xyz.shape[:-1] + (3, 3))

    # Rz
    Rz = np.stack([
        cos_z, -sin_z, zeros,
        sin_z,  cos_z, zeros,
        zeros,  zeros, ones,
    ], axis=-1).reshape(euler_xyz.shape[:-1] + (3, 3))

    # R = Rz @ Ry @ Rx  (extrinsic XYZ = intrinsic ZYX)
    return Rz @ Ry @ Rx


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to 6D representation (first two columns).

    Args:
        R: [..., 3, 3] rotation matrices.

    Returns:
        [..., 6] 6D rotation vectors: [R[:,0]; R[:,1]] concatenated.
    """
    # First two columns: R[..., :, 0] and R[..., :, 1]
    col0 = R[..., :, 0]  # [..., 3]
    col1 = R[..., :, 1]  # [..., 3]
    return np.concatenate([col0, col1], axis=-1)  # [..., 6]


# ── 7D → 10D action conversion ───────────────────────────────────────────────

def action_7d_to_10d(actions: np.ndarray) -> np.ndarray:
    """Convert 7D OXE actions to 10D representation compatible with dtw_nce.py.

    Args:
        actions: [..., 7] array. Layout: [xyz(3), euler_xyz(3), gripper(1)].

    Returns:
        [..., 10] array. Layout: [xyz(3), R6(6), gripper(1)].
    """
    xyz = actions[..., :3]                         # [..., 3]
    euler = actions[..., 3:6]                      # [..., 3]
    gripper = actions[..., 6:7]                    # [..., 1]

    R = euler_to_rotation_matrix(euler)            # [..., 3, 3]
    r6 = rotation_matrix_to_6d(R)                 # [..., 6]

    return np.concatenate([xyz, r6, gripper], axis=-1)  # [..., 10]


def pad_to_dim(a: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad the last dimension of *a* to *target_dim*.

    Args:
        a: Array with last dim <= target_dim.
        target_dim: Desired last dimension size.

    Returns:
        Zero-padded array with last dim == target_dim.
    """
    current_dim = a.shape[-1]
    if current_dim == target_dim:
        return a
    if current_dim > target_dim:
        raise ValueError(f"Array last dim {current_dim} > target_dim {target_dim}")
    pad_width = [(0, 0)] * (a.ndim - 1) + [(0, target_dim - current_dim)]
    return np.pad(a, pad_width, mode="constant", constant_values=0.0)
