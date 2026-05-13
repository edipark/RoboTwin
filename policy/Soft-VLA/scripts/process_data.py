"""Process raw RoboTwin HDF5 data into the format required for Soft-VLA fine-tuning.

Soft-VLA fine-tunes on **right-arm-only end-effector pose** actions, using a
6-D rotation representation (rot6d) instead of quaternions, plus a 1-D gripper
opening, for a total of **10-D** per timestep:

    [ right_xyz(3) + right_rot6d(6) + right_gripper(1) ]

This matches the canonical layout defined in
``src/openpi/policies/right_only_layout.py`` and is consumed by
``RoboTwinEEInputs`` / ``RoboTwinEEOutputs`` and the right-only execution path
in ``envs/_base_task.take_action(action_type='ee_right_10d')``.

Usage (from policy/Soft-VLA/):
    python scripts/process_data.py <task_name> <task_config> <expert_data_num>

Example:
    python scripts/process_data.py beat_block_hammer demo_clean 50

Input layout (relative to policy/Soft-VLA/):
    ../../data/<task_name>/<task_config>/
        data/episode{i}.hdf5
        instructions/episode{i}.json

Output layout:
    processed_data/<task_name>-<task_config>-<expert_data_num>/
        episode_{i}/
            episode_{i}.hdf5        # action (10-D right-only),
                                    # observations/{qpos(10-D), images}
            instructions.json       # {"instructions": [...]}
"""

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np

# ── Resolve openpi path so the canonical layout helpers are importable ──────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_POLICY_DIR  = os.path.dirname(_SCRIPTS_DIR)
_REPO_ROOT   = os.path.abspath(os.path.join(_POLICY_DIR, "..", "..", "..", ".."))
_OPENPI_SRC  = os.path.join(_REPO_ROOT, "src")
if _OPENPI_SRC not in sys.path:
    sys.path.insert(0, _OPENPI_SRC)

from openpi.policies.right_only_layout import (  # noqa: E402
    ACTION_FORMAT_TAG,
    ACTION_LAYOUT_TAG,
    RIGHT_ONLY_ACTION_DIM,
    pose_quat_to_rot6d_10d,
)


def load_hdf5(dataset_path):
    """Read end-effector pose action streams + images from one RoboTwin episode HDF5.

    Returns:
        right_endpose:   (T, 7)  — xyz + quat (w, x, y, z)
        right_gripper:   (T,)    — gripper opening in [0, 1]
        image_dict:      {cam_name: (T,) bytes-array of JPEG-encoded RGB frames}
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        if "/endpose" not in root:
            raise KeyError(
                f"{dataset_path} has no /endpose group. "
                "Re-collect the demo with `data_type.endpose: true` in the task config."
            )
        right_endpose = root["/endpose/right_endpose"][()]
        right_gripper = root["/endpose/right_gripper"][()]

        image_dict = dict()
        for cam_name in root["/observation/"].keys():
            if "rgb" in root[f"/observation/{cam_name}"]:
                image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return right_endpose, right_gripper, image_dict


def images_encoding(imgs):
    """Encode a list of RGB numpy arrays as JPEG bytes.

    cv2.imencode treats the input as BGR, so we convert RGB -> BGR first to
    ensure the stored JPEG has correct visual colors.  This makes the output
    compatible with the training-side decode_image_from_bytes pipeline which
    calls cv2.imdecode (returns BGR) + cv2.COLOR_BGR2RGB = correct RGB.
    """
    encode_data = []
    max_len = 0
    for i in range(len(imgs)):
        bgr_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode(".jpg", bgr_img)
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    padded_data = []
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def data_transform(path, episode_num, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    begin = 0
    for i in range(episode_num):
        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        ep_save_dir = os.path.join(save_path, f"episode_{i}")
        os.makedirs(ep_save_dir, exist_ok=True)

        with open(os.path.join(ep_save_dir, "instructions.json"), "w") as f:
            json.dump(save_instructions_json, f, indent=2)

        (
            right_endpose_all,
            right_gripper_all,
            image_dict,
        ) = load_hdf5(os.path.join(path, "data", f"episode{i}.hdf5"))

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []

        T = right_endpose_all.shape[0]
        for j in range(T):
            right_endpose = right_endpose_all[j]
            right_gripper = float(right_gripper_all[j])

            # 10-D right-only EE-pose vector with rot6d rotation.
            state = pose_quat_to_rot6d_10d(right_endpose, right_gripper)

            if j != T - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                # Raw HDF5 images are JPEG-encoded by pkl2hdf5 using cv2 (BGR convention).
                # Decode to BGR, then immediately convert to RGB so all stored images
                # are in RGB order — consistent with what the simulator renders and
                # what the training-side decode_image_from_bytes now expects.
                camera_high = cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                cam_high.append(cv2.resize(camera_high, (640, 480)))

                camera_right_bits = image_dict["right_camera"][j]
                camera_right = cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(camera_right_bits, np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                cam_right_wrist.append(cv2.resize(camera_right, (640, 480)))

                camera_left_bits = image_dict["left_camera"][j]
                camera_left = cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(camera_left_bits, np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                cam_left_wrist.append(cv2.resize(camera_left, (640, 480)))

            if j != 0:
                actions.append(state)

        qpos_arr    = np.array(qpos,    dtype=np.float32)   # [T-1, 10]
        actions_arr = np.array(actions, dtype=np.float32)   # [T-1, 10]
        # Store absolute xyz. Chunk-level delta (all horizon steps relative to
        # state_t) is applied at dataset load time in RoboTwinDataset.__getitem__,
        # matching DeltaActions in RoboTwinEEDataConfig (openpi/transforms.py).

        hdf5_path = os.path.join(ep_save_dir, f"episode_{i}.hdf5")
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("action", data=actions_arr)
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos_arr)
            f.attrs["action_format"] = ACTION_FORMAT_TAG
            f.attrs["action_layout"] = ACTION_LAYOUT_TAG
            f.attrs["action_dim"] = int(RIGHT_ONLY_ACTION_DIM)
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset(
                "cam_right_wrist", data=cam_right_enc, dtype=f"S{len_right}"
            )
            image.create_dataset(
                "cam_left_wrist", data=cam_left_enc, dtype=f"S{len_left}"
            )

        begin += 1
        print(f"process {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_name",
        type=str,
        help="Task name (e.g. beat_block_hammer)",
    )
    parser.add_argument("setting", type=str, help="Task setting (e.g. demo_clean)")
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", task_name, setting)
    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"

    print(f"Reading from: {os.path.abspath(load_dir)}")
    print(f"Saving to:    {os.path.abspath(target_dir)}")

    begin = data_transform(load_dir, expert_data_num, target_dir)
    print(f"\nDone. {begin} episodes processed → {target_dir}")
