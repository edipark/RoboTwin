"""Input / output transforms for RoboTwin demos using right-arm-only End-Effector
pose actions with a 6-D rotation representation.

Layout (matches the canonical right-only contract in
``openpi.policies.right_only_layout``):

    state  : [ right_xyz(3), right_rot6d(6), right_gripper(1) ]   -> 10-D
    action : same 10-D vector per timestep

Compared to legacy dual-arm 16-D EE layout:
- Single arm (right) only — left-arm execution is handled by ``_base_task``'s
  ``ee_right_10d`` action type, which holds the left arm in place.
- Rotation is rot6d (Zhou et al., 2019) instead of a quaternion.
- Image-key plumbing is unchanged: cam_high -> base_0_rgb,
  cam_left/right_wrist -> *_wrist_0_rgb.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.policies.right_only_layout import RIGHT_ONLY_ACTION_DIM


def make_robotwin_ee_example() -> dict:
    """Random input example matching the right-only 10-D EE-pose layout."""
    return {
        "state": np.zeros((RIGHT_ONLY_ACTION_DIM,), dtype=np.float32),
        "images": {
            "cam_high":        np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist":  np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _convert_image(img: np.ndarray) -> np.ndarray:
    """[C, H, W] uint8/float -> [H, W, C] uint8."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return einops.rearrange(img, "c h w -> h w c")


@dataclasses.dataclass(frozen=True)
class RoboTwinEEInputs(transforms.DataTransformFn):
    """Inputs for a RoboTwin right-only EE-pose policy.

    Expected inputs (matches what
    ``policy/Soft-VLA/softvla_model.update_observation_window`` produces, and
    what ``RoboTwinDataset.__getitem__`` produces during fine-tuning):
        - images: dict[name, img] where img is [channel, height, width].
                  Names must be a subset of EXPECTED_CAMERAS.
        - state:  [10] (right-only EE-pose layout above).
        - actions (training only): [action_horizon, 10].
        - prompt (optional): str.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unknown = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unknown:
            raise ValueError(
                f"Unexpected camera names {sorted(unknown)}; expected subset of {self.EXPECTED_CAMERAS}"
            )

        images = {name: _convert_image(img) for name, img in in_images.items()}

        if "cam_high" not in images:
            raise ValueError("RoboTwinEEInputs requires a 'cam_high' image.")
        base_image = images["cam_high"]

        out_images = {"base_0_rgb": base_image}
        out_masks = {"base_0_rgb": np.True_}

        for dest, source in (
            ("left_wrist_0_rgb",  "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if source in images:
                out_images[dest] = images[source]
                out_masks[dest]  = np.True_
            else:
                out_images[dest] = np.zeros_like(base_image)
                out_masks[dest]  = np.False_

        out: dict = {
            "image":      out_images,
            "image_mask": out_masks,
            "state":      np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        return out


@dataclasses.dataclass(frozen=True)
class RoboTwinEEOutputs(transforms.DataTransformFn):
    """Outputs for a RoboTwin right-only EE-pose policy.

    The model is configured with a padded action dim (e.g. 32). At inference we
    slice back to the real 10-D right-only EE-pose vector that
    ``_base_task.take_action(action_type='ee_right_10d')`` expects.
    """

    action_dim: int = RIGHT_ONLY_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., : self.action_dim]}


def _parse_image_hwc(image: np.ndarray) -> np.ndarray:
    """Return [H, W, C] uint8; handles both (C,H,W) float and (H,W,C) uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[1]:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# ── Step-delta integration helpers ────────────────────────────────────────────

def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation representation → 3×3 rotation matrix (Gram-Schmidt)."""
    a1 = rot6d[:3].astype(np.float64)
    a2 = rot6d[3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns are basis vectors


def _matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → 6D representation (first two columns)."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class StepDeltaIntegrateEEActions(transforms.DataTransformFn):
    """Integrate step-delta 10D EE actions back to absolute EE poses (inference only).

    The model predicts step-delta actions matching OXE/LIBERO pretraining format:
        delta_xyz(3)   = displacement from previous pose
        delta_rot6d(6) = 6D(R_{k-1}^T @ R_k), relative rotation (near identity)
        gripper(1)     = absolute gripper command (no integration needed)

    This transform recovers absolute EE poses suitable for robot execution by
    integrating deltas from the current observed state.
    """

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)

        if actions.ndim == 2:  # [H, 10]
            result = self._integrate_single(state.reshape(-1), actions)
        elif actions.ndim == 3:  # [B, H, 10]
            results = []
            for b in range(actions.shape[0]):
                s = state[b] if state.ndim == 2 else state
                results.append(self._integrate_single(s.reshape(-1), actions[b]))
            result = np.stack(results, axis=0)
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")

        data["actions"] = result.astype(np.float32)
        return data

    @staticmethod
    def _integrate_single(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Integrate step deltas for a single sample. actions: [H, 10]."""
        H = actions.shape[0]
        abs_actions = np.empty_like(actions)
        curr_xyz = state[:3].copy()
        R_curr = _rot6d_to_matrix(state[3:9])

        for k in range(H):
            delta_xyz = actions[k, :3]
            delta_rot6d = actions[k, 3:9]
            gripper = actions[k, 9]

            curr_xyz = curr_xyz + delta_xyz
            R_curr = R_curr @ _rot6d_to_matrix(delta_rot6d)

            abs_actions[k, :3] = curr_xyz
            abs_actions[k, 3:9] = _matrix_to_rot6d(R_curr)
            abs_actions[k, 9] = gripper

        return abs_actions


@dataclasses.dataclass(frozen=True)
class LeRobotRoboTwinEEInputs(transforms.DataTransformFn):
    """Inputs for LeRobot-format RoboTwin training (3 cameras, flat dataset keys).

    Used by LeRobotRoboTwinEEDataConfig after the RepackTransform has converted
    LeRobot's flat keys to observation/* keys:
        observation/image             — head / base camera     (H, W, 3)
        observation/wrist_image       — right wrist camera     (H, W, 3)
        observation/wrist_image_left  — left wrist camera      (H, W, 3)
        observation/state             — (10,) float32 absolute EE pose
        actions                       — (action_horizon, 10) float32  [training only]
        prompt                        — str  [optional]

    Wrist cameras are included automatically when ``observation/wrist_image`` and
    ``observation/wrist_image_left`` are present in ``data``. Omit them from the
    upstream RepackTransform to train with the head camera only.
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image_hwc(data["observation/image"])

        image: dict = {"base_0_rgb": base_image}
        image_mask: dict = {"base_0_rgb": np.True_}

        if "observation/wrist_image" in data:
            image["right_wrist_0_rgb"] = _parse_image_hwc(data["observation/wrist_image"])
            image_mask["right_wrist_0_rgb"] = np.True_
        if "observation/wrist_image_left" in data:
            image["left_wrist_0_rgb"]  = _parse_image_hwc(data["observation/wrist_image_left"])
            image_mask["left_wrist_0_rgb"]  = np.True_

        out: dict = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": image,
            "image_mask": image_mask,
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        return out
