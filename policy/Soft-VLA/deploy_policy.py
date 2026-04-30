"""RoboTwin deploy_policy interface for Soft-VLA (right-arm-only 10-D actions).

Follows the same structure as policy/pi05/deploy_policy.py.

State / action layout (matches scripts/process_data.py output):

    [ right_xyz(3), right_rot6d(6), right_gripper(1) ]   # 10-D total

Eval execution dispatches each predicted action through
``_base_task.take_action(action, action_type='ee_right_10d')``, which decodes
rot6d back into a quaternion, plans the right arm, and holds the left arm in
place.
"""

import os
import sys

import numpy as np

# ── make the Soft-VLA source accessible ───────────────────────────────────────
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_POLICY_DIR, "src")
if _POLICY_DIR not in sys.path:
    sys.path.insert(0, _POLICY_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from softvla_model import SoftVLA  # noqa: E402

# Local rot6d helpers (RoboTwin envs/utils mirror of openpi.policies.right_only_layout).
from envs.utils.rot6d import (  # noqa: E402
    RIGHT_ONLY_ACTION_DIM,
    pose_quat_to_rot6d_10d,
)


# ── observation encoding ───────────────────────────────────────────────────────

def _ee_state_from_obs(observation: dict) -> np.ndarray:
    """Build the 10-D right-only EE-pose state vector from a RoboTwin observation.

    Layout: [right_xyz(3), right_rot6d(6), right_gripper(1)].
    Matches the dataset action layout produced by ``scripts/process_data.py`` and
    the slicing performed by ``_base_task.take_action(action_type='ee_right_10d')``.
    """
    if "endpose" not in observation:
        raise KeyError(
            "RoboTwin observation has no 'endpose' field. "
            "Set `data_type.endpose: true` in the task config."
        )
    ep = observation["endpose"]
    right_endpose = np.asarray(ep["right_endpose"], dtype=np.float32).reshape(-1)
    right_gripper = float(np.asarray(ep["right_gripper"]).reshape(-1)[0])
    return pose_quat_to_rot6d_10d(right_endpose, right_gripper)


def encode_obs(observation: dict) -> tuple:
    """Extract camera images and right-only EE-pose state from a RoboTwin observation dict.

    Returns:
        input_rgb_arr: list of three HWC uint8 numpy arrays
                       [head/front, right wrist, left wrist]
        input_state:   1-D float numpy array, 10-D right-only EE-pose vector.
    """
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = _ee_state_from_obs(observation)
    return input_rgb_arr, input_state


# ── model construction ─────────────────────────────────────────────────────────

def get_model(usr_args: dict) -> SoftVLA:
    """Instantiate and return a SoftVLA policy model.

    Args:
        usr_args: dict of YAML / CLI override values.  Expected keys:
            train_config_name  (str)  – TrainConfig name in openpi/training/config.py
            model_name         (str)  – checkpoint experiment name
            checkpoint_id      (int)  – checkpoint step
            softvla_step       (int)  – number of actions to execute per inference call
                                        (action chunk execution length, like pi0_step).
                                        E.g. softvla_step=8 runs only the first 8 predicted actions.
            num_denoise_steps  (int, optional) – flow-matching ODE integration steps for
                                        action generation (default 10). Higher = better quality
                                        but slower. Separate from softvla_step.
            domain_id          (int, optional) – embodiment index for soft prompt (default 0)
    """
    return SoftVLA(
        train_config_name=usr_args["train_config_name"],
        model_name=usr_args["model_name"],
        checkpoint_id=int(usr_args["checkpoint_id"]),
        softvla_step=int(usr_args["softvla_step"]),
        num_denoise_steps=int(usr_args.get("num_denoise_steps", 10)),
        domain_id=int(usr_args.get("domain_id", 0)),
    )


# ── evaluation loop ────────────────────────────────────────────────────────────

def eval(TASK_ENV, model: SoftVLA, observation: dict) -> None:
    """Run one action-chunk evaluation step.

    At the first call (observation_window is None), initialises the language
    instruction.  For each subsequent call the observation window is updated
    before inference.
    """
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ── get action chunk ────────────────────────────────────────────────────
    actions = model.get_action()[: model.softvla_step]

    # ── execute each action step (right-only EE: xyz + rot6d + gripper) ─────
    for action in actions:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < RIGHT_ONLY_ACTION_DIM:
            raise ValueError(
                f"Predicted action has {action.shape[0]} dims, expected "
                f">= {RIGHT_ONLY_ACTION_DIM} for ee_right_10d execution."
            )
        TASK_ENV.take_action(action[:RIGHT_ONLY_ACTION_DIM], action_type="ee_right_10d")
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


# ── episode reset ──────────────────────────────────────────────────────────────

def reset_model(model: SoftVLA) -> None:
    """Clear observation cache and instruction at the beginning of each episode."""
    model.reset_observation_windows()
