"""Canonical right-arm-only 10-D action layout for Soft-VLA RoboTwin pipeline.

Layout (single right arm only):

    state / action  =  [ x, y, z,                 # 3 - position
                         r0, r1, r2, r3, r4, r5,  # 6 - rot6d (first two columns of R)
                         gripper ]                # 1 - gripper opening in [0, 1]

`rot6d` follows the "continuity of rotation representations" parameterisation
(Zhou et al., 2019).  We pack the FIRST TWO COLUMNS of the rotation matrix R
(``R[:, 0]`` then ``R[:, 1]``) into a flat 6-vector.

Quaternion convention is the transforms3d / sapien default ``(w, x, y, z)``.

Helpers in this module are pure-numpy and self-contained so they can be used
from training-side code (the ``.venv`` Python) and from inference-side code
(the RoboTwin Python which already imports ``openpi``).  The RoboTwin env
(_base_task.py) which lives outside the openpi package uses an identical
mirror in ``envs/utils/rot6d.py`` to avoid an inbound dependency on openpi.
"""

from __future__ import annotations

import numpy as np


# ── Public constants ────────────────────────────────────────────────────────

RIGHT_ONLY_ACTION_DIM: int = 10
"""Right-arm-only action / state dimension."""

POSE_QUAT_DIM: int = 7
"""xyz(3) + quat(4) — RoboTwin's raw EE-pose representation."""

ACTION_FORMAT_TAG: str = "ee_right_only_rot6d_10d_dxyz"
"""Stored on processed HDF5 ``action_format`` attr to identify the new layout.
xyz(3) are delta w.r.t. current state; rot6d(6) and gripper(1) are absolute."""

ACTION_LAYOUT_TAG: str = (
    "delta_right_xyz(3) + right_rot6d(6) + right_gripper(1)"
)


# ── Quaternion <-> rot6d ────────────────────────────────────────────────────

def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` quaternion → 3x3 rotation matrix (numpy-only)."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quat must have 4 elements, got {q.shape[0]}")
    n = float(np.dot(q, q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q * np.sqrt(2.0 / n)
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - (yy + zz),       xy - wz,        xz + wy],
            [      xy + wz,   1.0 - (xx + zz),      yz - wx],
            [      xz - wy,         yz + wx,  1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → ``(w, x, y, z)`` quaternion (numpy-only)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    return q


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` quat → rot6d (6,) = first two columns of rotation matrix."""
    R = _quat_to_mat(quat)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def rot6d_to_rot_mat(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (6,) → 3x3 rotation matrix via Gram–Schmidt orthogonalisation."""
    v = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    if v.shape[0] != 6:
        raise ValueError(f"rot6d must have 6 elements, got {v.shape[0]}")
    a1, a2 = v[:3], v[3:]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        b1 = np.array([1.0, 0.0, 0.0])
    else:
        b1 = a1 / n1
    a2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2)
    if n2 < 1e-8:
        # Fall back to a vector orthogonal to b1.
        helper = np.array([0.0, 1.0, 0.0]) if abs(b1[0]) < 0.9 else np.array([1.0, 0.0, 0.0])
        a2 = helper - np.dot(b1, helper) * b1
        n2 = np.linalg.norm(a2)
    b2 = a2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (6,) → ``(w, x, y, z)`` quaternion."""
    R = rot6d_to_rot_mat(rot6d)
    return _mat_to_quat(R).astype(np.float32)


# ── Pose composition helpers ────────────────────────────────────────────────

def pose_quat_to_rot6d_10d(pose7: np.ndarray, gripper: float) -> np.ndarray:
    """Convert RoboTwin's 7-D EE pose (xyz + quat) plus a scalar gripper opening
    into the canonical 10-D right-only action layout.
    """
    pose7 = np.asarray(pose7, dtype=np.float32).reshape(-1)
    if pose7.shape[0] != POSE_QUAT_DIM:
        raise ValueError(f"pose7 must have {POSE_QUAT_DIM} elements, got {pose7.shape[0]}")
    xyz = pose7[:3]
    quat = pose7[3:7]
    rot6d = quat_to_rot6d(quat)
    return np.concatenate(
        [xyz.astype(np.float32), rot6d.astype(np.float32), np.float32([gripper])],
        dtype=np.float32,
    )


def rot6d_10d_to_pose_quat(action10: np.ndarray) -> tuple[np.ndarray, float]:
    """Inverse of ``pose_quat_to_rot6d_10d``.

    Returns:
        pose7  : float32 array of shape (7,) = xyz + (w, x, y, z) quat.
        gripper: float scalar in [0, 1] (caller should clip if needed).
    """
    a = np.asarray(action10, dtype=np.float32).reshape(-1)
    if a.shape[0] != RIGHT_ONLY_ACTION_DIM:
        raise ValueError(
            f"action must have {RIGHT_ONLY_ACTION_DIM} elements, got {a.shape[0]}"
        )
    xyz = a[:3]
    rot6d = a[3:9]
    gripper = float(a[9])
    quat = rot6d_to_quat(rot6d)
    pose7 = np.concatenate([xyz, quat], dtype=np.float32)
    return pose7, gripper
