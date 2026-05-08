"""OXE dataset registry for Soft-VLA.

Maps each dataset folder (= tfds dataset name) to its version, robot domain,
and observation keys.  Domain IDs group datasets by robot platform so the model
can learn per-robot soft prompts in Phase 2.

Domain grouping (NUM_ROBOTS = 8):
  0 - Franka + Default Gripper    (buds, sailor, sirius, mutex, furniture_bench, viola, fmb)
  1 - Franka + Robotiq 2F-85      (droid)
  2 - Franka + Custom 3D Gripper  (taco_play)
  3 - Google Robot                (bc_z, fractal20220817_data)
  4 - UR5 + Robotiq 2F-85         (berkeley_autolab_ur5)
  5 - Fanuc Mate                  (berkeley_fanuc_manipulation)
  6 - Jaco 2                      (jaco_play)
  7 - WidowX                      (bridge)
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset RLDS configuration.

    Attributes:
        tfds_name:            Dataset name as registered in tensorflow_datasets.
        version:              TFDS version string to load (e.g. "0.1.0").
        domain_id:            Integer robot-platform ID (0 ... NUM_ROBOTS-1).
        weight:               Mixing weight (all configs must sum to 1.0).
        action_keys:          Tuple of slash-separated RLDS nested paths that
                              will be looked up and concatenated on the last axis
                              to form the action tensor.  For most datasets a
                              single ("action",) suffices; datasets like fractal
                              use multiple sub-keys.
        action_key_max_dim:   For each key in action_keys, maximum number of
                              dimensions to take from the last axis.  None = all.
                              Used for datasets like bc_z where the action has
                              future predictions bundled in a wide vector.
        primary_image_key:    Slash-separated path for the main exterior camera.
        wrist_image_key:      Slash-separated path for the wrist camera, or None.
        state_key:            Slash-separated path for low-dimensional robot state.
        state_dim:            Actual dimension of the state vector in this dataset.
        language_key:         Slash-separated path to the natural-language
                              instruction string tensor.
    """
    tfds_name: str
    version: str
    domain_id: int
    weight: float
    # Action construction
    action_keys: tuple = ("action",)
    action_key_max_dim: tuple = (None,)
    # Image keys
    primary_image_key: str = "observation/image"
    wrist_image_key: str | None = "observation/wrist_image"
    # State
    state_key: str = "observation/state"
    state_dim: int = 7
    # Language
    language_key: str = "language_instruction"

    def __post_init__(self) -> None:
        action_keys = tuple(self.action_keys)
        action_key_max_dim = tuple(self.action_key_max_dim)
        if len(action_key_max_dim) == 1 and len(action_keys) > 1:
            action_key_max_dim = action_key_max_dim * len(action_keys)
        if len(action_keys) != len(action_key_max_dim):
            raise ValueError(
                "action_keys and action_key_max_dim must have the same length: "
                f"tfds_name={self.tfds_name} action_keys={action_keys} action_key_max_dim={action_key_max_dim}"
            )
        object.__setattr__(self, "action_keys", action_keys)
        object.__setattr__(self, "action_key_max_dim", action_key_max_dim)


# ---------------------------------------------------------------------------
# Registry
# Domain IDs are assigned per robot hardware platform (arm + gripper type).
# 15 datasets, equal mixing weights.
# ---------------------------------------------------------------------------

_N = 15
_W = 1.0 / _N

DATASET_REGISTRY: list = [
    # Domain 0 -- Franka + Default Gripper
    DatasetConfig(
        tfds_name="austin_buds_dataset_converted_externally_to_rlds",
        version="0.1.0",
        domain_id=0,
        weight=_W,
        state_dim=24,
    ),
    DatasetConfig(
        tfds_name="austin_sailor_dataset_converted_externally_to_rlds",
        version="0.1.0",
        domain_id=0,
        weight=_W,
        state_dim=8,
    ),
    DatasetConfig(
        tfds_name="austin_sirius_dataset_converted_externally_to_rlds",
        version="0.1.0",
        domain_id=0,
        weight=_W,
        state_dim=8,
    ),
    DatasetConfig(
        tfds_name="utaustin_mutex",
        version="0.1.0",
        domain_id=0,
        weight=_W,
        state_dim=24,
    ),
    # action (T,8), observation/state (T,35)
    DatasetConfig(
        tfds_name="furniture_bench_dataset_converted_externally_to_rlds",
        version="0.1.0",
        domain_id=0,
        weight=_W,
        state_dim=35,
    ),
    # world_vector(3)+rotation_delta(3)+gripper(scalar) -> 7D
    DatasetConfig(
        tfds_name="viola",
        version="0.1.0",
        domain_id=0,
        weight=_W,
        action_keys=(
            "action/world_vector",
            "action/rotation_delta",
            "action/gripper_closedness_action",
        ),
        primary_image_key="observation/agentview_rgb",
        wrist_image_key="observation/eye_in_hand_rgb",
        state_key="observation/joint_states",
        state_dim=7,
        language_key="observation/natural_language_instruction",
    ),
    # action (T,7), image_side_1 / image_wrist_1, eef_pose (7D)
    DatasetConfig(
        tfds_name="fmb",
        version="0.0.1",
        domain_id=0,
        weight=_W,
        primary_image_key="observation/image_side_1",
        wrist_image_key="observation/image_wrist_1",
        state_key="observation/eef_pose",
        state_dim=7,
    ),
    # Domain 1 -- Franka + Robotiq 2F-85 Gripper
    # action (T,7) float64, cast to float32 in restructure
    DatasetConfig(
        tfds_name="droid",
        version="1.0.1",
        domain_id=1,
        weight=_W,
        primary_image_key="observation/exterior_image_1_left",
        wrist_image_key="observation/wrist_image_left",
        state_key="observation/joint_position",
        state_dim=7,
    ),
    # Domain 2 -- Franka + Custom 3D-Printed Gripper
    # action/actions (T,7), observation/robot_obs (T,15)
    DatasetConfig(
        tfds_name="taco_play",
        version="0.1.0",
        domain_id=2,
        weight=_W,
        action_keys=("action/actions",),
        primary_image_key="observation/rgb_static",
        wrist_image_key="observation/rgb_gripper",
        state_key="observation/robot_obs",
        state_dim=15,
        language_key="observation/natural_language_instruction",
    ),
    # Domain 3 -- Google Robot
    # bc_z: action is future predictions; take first 3/3/1 dims -> 7D
    DatasetConfig(
        tfds_name="bc_z",
        version="0.1.0",
        domain_id=3,
        weight=_W,
        action_keys=(
            "action/future/xyz_residual",
            "action/future/axis_angle_residual",
            "action/future/target_close",
        ),
        action_key_max_dim=(3, 3, 1),
        wrist_image_key=None,
        state_key="observation/present/xyz",
        state_dim=3,
        language_key="observation/natural_language_instruction",
    ),
    # fractal: world_vector(3)+rotation_delta(3)+gripper(1) -> 7D
    DatasetConfig(
        tfds_name="fractal20220817_data",
        version="0.1.0",
        domain_id=3,
        weight=_W,
        action_keys=(
            "action/world_vector",
            "action/rotation_delta",
            "action/gripper_closedness_action",
        ),
        wrist_image_key=None,
        state_key="observation/base_pose_tool_reached",
        state_dim=7,
        language_key="observation/natural_language_instruction",
    ),
    # Domain 4 -- UR5 + Robotiq 2F-85 Gripper
    # world_vector(3)+rotation_delta(3)+gripper(scalar) -> 7D
    DatasetConfig(
        tfds_name="berkeley_autolab_ur5",
        version="0.1.0",
        domain_id=4,
        weight=_W,
        action_keys=(
            "action/world_vector",
            "action/rotation_delta",
            "action/gripper_closedness_action",
        ),
        wrist_image_key="observation/hand_image",
        state_key="observation/robot_state",
        state_dim=15,
        language_key="observation/natural_language_instruction",
    ),
    # Domain 5 -- Fanuc Mate
    # action (T,6), observation/state (T,13)
    DatasetConfig(
        tfds_name="berkeley_fanuc_manipulation",
        version="0.1.0",
        domain_id=5,
        weight=_W,
        state_dim=13,
    ),
    # Domain 6 -- Jaco 2
    # action/world_vector(3)+gripper(1) -> 4D (padded to 7 in model)
    DatasetConfig(
        tfds_name="jaco_play",
        version="0.1.0",
        domain_id=6,
        weight=_W,
        action_keys=(
            "action/world_vector",
            "action/gripper_closedness_action",
        ),
        wrist_image_key="observation/image_wrist",
        state_key="observation/end_effector_cartesian_pos",
        state_dim=7,
        language_key="observation/natural_language_instruction",
    ),
    # Domain 7 -- WidowX
    # action: world_vector(3)+rotation_delta(3)+open_gripper(bool→float32, 1) -> 7D
    # open_gripper is bool dtype; tf.cast(t, tf.float32) in restructure handles it.
    DatasetConfig(
        tfds_name="bridge",
        version="0.1.0",
        domain_id=7,
        weight=_W,
        action_keys=(
            "action/world_vector",
            "action/rotation_delta",
            "action/open_gripper",
        ),
        action_key_max_dim=(3, 3, 1),
        wrist_image_key=None,
        state_key="observation/state",
        state_dim=7,
        language_key="observation/natural_language_instruction",
    ),
]

# Sanity-check
_weight_sum = sum(c.weight for c in DATASET_REGISTRY)
assert abs(_weight_sum - 1.0) < 1e-6, f"Dataset weights sum to {_weight_sum}, expected 1.0"

# Number of distinct robot platforms
NUM_ROBOTS: int = len({c.domain_id for c in DATASET_REGISTRY})

# Legacy alias (used in some parts of the codebase)
OXE_ACTION_KEY: str = "action"

# String constants for restructured batch dict keys.
# Used by precompute scripts and any code that reads batches from the pipeline.
ACTION: str = "action"
DOMAIN_ID: str = "domain_id"
