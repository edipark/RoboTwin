"""RoboTwin deploy_policy interface for Soft-VLA (right-arm-only 10-D actions).

.. note::
   This module is imported in **both** environments:

   * The RoboTwin conda env (has sapien, no flax) – runs the simulator client.
   * The Soft-VLA venv (has flax, torch, softvla) – runs the policy server.

   All heavy imports (``softvla_model``, which chains to flax) are therefore
   **lazy** – deferred to the functions that actually need them so that a bare
   ``import deploy_policy`` from the RoboTwin conda env succeeds cleanly.

Follows the same structure as policy/pi05/deploy_policy.py.

State / action layout (matches scripts/process_data.py output):

    [ right_xyz(3), right_rot6d(6), right_gripper(1) ]   # 10-D total

Eval execution dispatches each predicted action through
``_base_task.take_action(action, action_type='ee_right_10d')``, which decodes
rot6d back into a quaternion, plans the right arm, and holds the left arm in
place.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ── make the Soft-VLA source accessible ───────────────────────────────────────
# NOTE: softvla_model (and hence flax) is intentionally NOT imported here at
# module level so this file can be imported in the RoboTwin conda env.
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_POLICY_DIR, "src")
if _POLICY_DIR not in sys.path:
    sys.path.insert(0, _POLICY_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# NOTE: rot6d helpers (RIGHT_ONLY_ACTION_DIM, pose_quat_to_rot6d_10d) are
# imported lazily inside the functions that use them (_ee_state_from_obs, eval)
# so this file can be imported in the Soft-VLA venv without transforms3d/sapien.


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
    from envs.utils.rot6d import pose_quat_to_rot6d_10d  # lazy: needs transforms3d (conda only)
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
    from softvla_model import SoftVLA  # lazy: only needed in venv (server side)
    return SoftVLA(
        train_config_name=usr_args["train_config_name"],
        model_name=usr_args["model_name"],
        checkpoint_id=int(usr_args["checkpoint_id"]),
        softvla_step=int(usr_args["softvla_step"]),
        num_denoise_steps=int(usr_args.get("num_denoise_steps", 10)),
        domain_id=int(usr_args.get("domain_id", 0)),
        num_robots=int(usr_args.get("num_robots", 8)),
    )


# ── evaluation loop ────────────────────────────────────────────────────────────

def eval(TASK_ENV, model, observation: dict) -> None:  # noqa: A001
    """Run one action-chunk evaluation step.

    Works in two modes:

    * **Single-process** (``model`` is a ``SoftVLA`` instance): direct attribute
      access — used by ``eval.sh``.
    * **Dual-env** (``model`` is a ``ModelClient`` with a ``.call()`` method):
      all inference is routed through the socket to the policy server — used
      by ``eval_double_env.sh``.
    """
    input_rgb_arr, input_state = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    if hasattr(model, "call"):  # ── dual-env: delegate to policy server via socket
        obs_packed = {
            "img_arr": input_rgb_arr,
            "state": input_state,
            "instruction": instruction,
        }
        result = model.call("eval_step", obs=obs_packed)
        actions = np.asarray(result["actions"])
        softvla_step = int(result["softvla_step"])
    else:  # ── single-process: direct SoftVLA access
        if model.observation_window is None:
            model.set_language(instruction)
        model.update_observation_window(input_rgb_arr, input_state)
        actions = model.get_action()
        softvla_step = model.softvla_step

    # ── execute each action step (right-only EE: xyz + rot6d + gripper) ─────
    from envs.utils.rot6d import RIGHT_ONLY_ACTION_DIM  # lazy: needs transforms3d (conda only)
    # Debug: print first action to verify delta→absolute conversion
    import logging as _logging
    _dbg = _logging.getLogger(__name__)
    _dbg.info("[eval] state_xyz=%.4f,%.4f,%.4f | action[0]_xyz=%.4f,%.4f,%.4f | chunk_shape=%s",
              input_state[0], input_state[1], input_state[2],
              actions[0, 0], actions[0, 1], actions[0, 2],
              actions.shape)
    for action in actions[:softvla_step]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < RIGHT_ONLY_ACTION_DIM:
            raise ValueError(
                f"Predicted action has {action.shape[0]} dims, expected "
                f">= {RIGHT_ONLY_ACTION_DIM} for ee_right_10d execution."
            )
        TASK_ENV.take_action(action[:RIGHT_ONLY_ACTION_DIM], action_type="ee_right_10d")
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        if not hasattr(model, "call"):  # single-process: update obs window in-loop
            model.update_observation_window(input_rgb_arr, input_state)


# ── episode reset ──────────────────────────────────────────────────────────────

def reset_model(model) -> None:
    """Clear observation cache and instruction at the beginning of each episode."""
    model.reset_observation_windows()
