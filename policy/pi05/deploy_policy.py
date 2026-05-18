import numpy as np
import torch
import dill
import os, sys
import transforms3d as t3d

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


def _quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] → rot6d (first two columns of rotation matrix, 6-D)."""
    rot = t3d.quaternions.quat2mat(quat)  # 3×3
    return np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32)


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]

    # Build 10-D state: [right_xyz(3), right_rot6d(6), right_gripper(1)]
    endpose      = observation["endpose"]
    right_ee     = endpose["right_endpose"]   # [x, y, z, qw, qx, qy, qz]
    right_gripper = float(endpose["right_gripper"])

    xyz   = np.array(right_ee[:3], dtype=np.float32)
    quat  = np.array(right_ee[3:], dtype=np.float32)  # [qw, qx, qy, qz]
    rot6d = _quat_to_rot6d(quat)
    input_state = np.concatenate([xyz, rot6d, [right_gripper]])

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action, action_type='ee_right_10d')
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
