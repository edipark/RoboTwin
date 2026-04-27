"""RoboTwin deploy_policy interface for Soft-VLA.

Follows the same structure as policy/pi05/deploy_policy.py.
"""

import os
import sys

import numpy as np

# ── make the Soft-VLA source accessible ───────────────────────────────────────
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_POLICY_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from softvla_model import SoftVLA  # noqa: E402


# ── observation encoding ───────────────────────────────────────────────────────

def encode_obs(observation: dict) -> tuple:
    """Extract camera images and joint state from a RoboTwin observation dict.

    Returns:
        input_rgb_arr: list of three HWC uint8 numpy arrays
                       [head/front, right wrist, left wrist]
        input_state:   1-D float numpy array of joint positions
    """
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


# ── model construction ─────────────────────────────────────────────────────────

def get_model(usr_args: dict) -> SoftVLA:
    """Instantiate and return a SoftVLA policy model.

    Args:
        usr_args: dict of YAML / CLI override values.  Expected keys:
            train_config_name  (str)  – TrainConfig name in openpi/training/config.py
            model_name         (str)  – checkpoint experiment name
            checkpoint_id      (int)  – checkpoint step
            softvla_step       (int)  – flow-matching denoising steps to execute
            domain_id          (int, optional) – embodiment index for soft prompt (default 0)
    """
    return SoftVLA(
        train_config_name=usr_args["train_config_name"],
        model_name=usr_args["model_name"],
        checkpoint_id=int(usr_args["checkpoint_id"]),
        softvla_step=int(usr_args["softvla_step"]),
        domain_id=int(usr_args.get("domain_id", 0)),
    )


# ── evaluation loop ────────────────────────────────────────────────────────────

def eval(TASK_ENV, model: SoftVLA, observation: dict) -> None:
    """Run one action-chunk evaluation step.

    At the first call (observation_window is None), initialises the language
    instruction.  For each subsequent call the observation window is updated
    before inference.
    """
    # Initialise language instruction on first frame of each episode.
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ── get action chunk ────────────────────────────────────────────────────
    actions = model.get_action()[: model.softvla_step]

    # ── execute each action step ────────────────────────────────────────────
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


# ── episode reset ──────────────────────────────────────────────────────────────

def reset_model(model: SoftVLA) -> None:
    """Clear observation cache and instruction at the beginning of each episode."""
    model.reset_observation_windows()
