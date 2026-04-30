"""Rot6d helpers and right-only 10-D action layout (RoboTwin-side mirror).

This module mirrors the contract defined in
``src/openpi/policies/right_only_layout.py`` so the RoboTwin simulator core
(``envs/_base_task.py``) can decode 10-D right-only actions emitted by the
Soft-VLA policy without taking a hard dependency on ``openpi``.

Layout (single right arm only):

    [ x, y, z, r0, r1, r2, r3, r4, r5, gripper ]   # 10-D total

    rot6d  = first two columns of rotation matrix R, flattened in column-major
             order: ``[R[:,0], R[:,1]]``.
    quat   = transforms3d / sapien convention ``(w, x, y, z)``.

Keep this file numpy-only to avoid additional dependencies in the RoboTwin
runtime environment.
"""

from __future__ import annotations

import numpy as np
import transforms3d as t3d


RIGHT_ONLY_ACTION_DIM: int = 10
POSE_QUAT_DIM: int = 7


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` quat → rot6d (6,) = first two columns of rotation matrix."""
    R = t3d.quaternions.quat2mat(np.asarray(quat).reshape(-1))
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def rot6d_to_rot_mat(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (6,) → 3x3 rotation matrix via Gram-Schmidt orthogonalisation."""
    v = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    if v.shape[0] != 6:
        raise ValueError(f"rot6d must have 6 elements, got {v.shape[0]}")
    a1, a2 = v[:3], v[3:]
    n1 = float(np.linalg.norm(a1))
    if n1 < 1e-8:
        b1 = np.array([1.0, 0.0, 0.0])
    else:
        b1 = a1 / n1
    a2 = a2 - np.dot(b1, a2) * b1
    n2 = float(np.linalg.norm(a2))
    if n2 < 1e-8:
        helper = np.array([0.0, 1.0, 0.0]) if abs(b1[0]) < 0.9 else np.array([1.0, 0.0, 0.0])
        a2 = helper - np.dot(b1, helper) * b1
        n2 = float(np.linalg.norm(a2))
    b2 = a2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (6,) → ``(w, x, y, z)`` quaternion."""
    R = rot6d_to_rot_mat(rot6d)
    q = t3d.quaternions.mat2quat(R)
    return np.asarray(q, dtype=np.float32)


def rot6d_10d_to_pose_quat(action10: np.ndarray) -> tuple[np.ndarray, float]:
    """10-D right-only action → (7-D xyz+quat pose, gripper scalar).

    Inverse of the encoding produced by ``process_data.py`` /
    ``deploy_policy._ee_state_from_obs``.
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


def pose_quat_to_rot6d_10d(pose7: np.ndarray, gripper: float) -> np.ndarray:
    """7-D xyz+quat pose plus scalar gripper → 10-D right-only action layout."""
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
