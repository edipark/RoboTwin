"""PyTorch Dataset for processed RoboTwin data (Soft-VLA Phase-2 fine-tuning).

Reads the output of scripts/process_data.py:
    processed_data/<task>-<config>-<N>/
        episode_{i}/
            episode_{i}.hdf5   (action, observations/{qpos, images/{cam_*}})
            instructions.json  ({"instructions": [...]})

Each __getitem__ returns one (state, action-chunk, images, tokens, domain_id) sample.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ── Resolve Soft-VLA src (policy/Soft-VLA/src → Soft-VLA/src via symlink) ──
_POLICY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_POLICY_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from openpi.models.tokenizer import PaligemmaTokenizer  # noqa: E402
from openpi.shared import normalize as _normalize        # noqa: E402


class RoboTwinDataset(Dataset):
    """Dataset over processed RoboTwin episodes for Soft-VLA Phase-2 fine-tuning.

    Args:
        processed_dir:  Path to processed_data/<task>-<config>-<N>/ directory.
        action_horizon: Number of future actions per sample (chunk length).
                        Must match the pre-trained SoftVLA model's action_horizon.
        action_dim:     Model-side action / state dimensionality (typically 32).
                        Must match the pre-trained SoftVLA model's action_dim.
                        Real right-only EE actions on disk are 10-D
                        (xyz + rot6d + gripper) — they are zero-padded up to
                        ``action_dim`` here, and sliced back down by
                        ``RoboTwinEEOutputs`` at inference.
        max_token_len:  Maximum token length for the PaliGemma tokenizer.
        norm_stats:     Optional dict with keys "state" and/or "actions"
                        (plural — matches the upstream OpenPI convention used by
                        Policy.infer's output dict and Unnormalize transform).
                        Each value is a openpi.shared.normalize.NormStats instance.
                        If None, no normalization is applied.
        domain_id:      Embodiment index for SoftPromptHub (passed through to model).
        image_size:     Target H=W for resized images (pixels).
    """

    def __init__(
        self,
        processed_dir: str,
        action_horizon: int = 16,
        action_dim: int = 32,
        max_token_len: int = 200,
        norm_stats: dict | None = None,
        domain_id: int = 0,
        image_size: int = 224,
    ):
        self.processed_dir = processed_dir
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.max_token_len = max_token_len
        self.norm_stats = norm_stats
        self.domain_id = domain_id
        self.image_size = image_size

        # ── Build (ep_dir, timestep) index ────────────────────────────────
        self._index: list[tuple[str, int]] = []

        ep_dirs = sorted(
            [
                d
                for d in os.listdir(processed_dir)
                if os.path.isdir(os.path.join(processed_dir, d))
                and d.startswith("episode_")
            ],
            key=lambda x: int(x.split("_")[1]),
        )

        if not ep_dirs:
            raise ValueError(f"No episode_ directories found in {processed_dir}")

        for ep_dir in ep_dirs:
            hdf5_path = os.path.join(processed_dir, ep_dir, f"{ep_dir}.hdf5")
            if not os.path.isfile(hdf5_path):
                continue
            with h5py.File(hdf5_path, "r") as f:
                T = f["action"].shape[0]
            for t in range(T):
                self._index.append((ep_dir, t))

        if not self._index:
            raise ValueError(f"No valid HDF5 files found in {processed_dir}")

        # ── Tokenizer (Pi05 mode: state embedded in text prompt) ──────────
        self.tokenizer = PaligemmaTokenizer(max_len=max_token_len)

        # ── Per-episode in-memory cache ───────────────────────────────────
        # Typical fine-tuning uses 50-200 episodes; full cache is safe.
        self._cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self._index)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _load_episode(self, ep_dir: str) -> dict:
        """Load and cache all arrays for one episode."""
        if ep_dir in self._cache:
            return self._cache[ep_dir]

        ep_path = os.path.join(self.processed_dir, ep_dir)
        hdf5_path = os.path.join(ep_path, f"{ep_dir}.hdf5")
        instr_path = os.path.join(ep_path, "instructions.json")

        with h5py.File(hdf5_path, "r") as f:
            actions = f["action"][()].astype(np.float32)          # [T, D]
            qpos    = f["observations/qpos"][()].astype(np.float32)  # [T, D]
            cam_high        = f["observations/images/cam_high"][()]
            cam_right_wrist = f["observations/images/cam_right_wrist"][()]
            cam_left_wrist  = f["observations/images/cam_left_wrist"][()]

        with open(instr_path, "r") as f:
            instr_dict = json.load(f)
        instructions = instr_dict.get("instructions", [""])

        ep = {
            "actions":         actions,
            "qpos":            qpos,
            "cam_high":        cam_high,
            "cam_right_wrist": cam_right_wrist,
            "cam_left_wrist":  cam_left_wrist,
            "instructions":    instructions,
        }
        self._cache[ep_dir] = ep
        return ep

    def _decode_image(self, jpeg_bytes) -> np.ndarray:
        """JPEG bytes → RGB float32 [H, W, 3] in [-1, 1] range, resized to image_size.

        preprocessing_pytorch.py expects images in [-1, 1] (it converts to [0, 1] internally
        for augmentations by doing `image / 2 + 0.5`).  softvla_model.py's _prepare_image
        also normalises to [-1, 1], so train and eval are consistent.
        """
        arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img.astype(np.float32) / 127.5 - 1.0  # [0, 255] → [-1, 1]

    # ── __getitem__ ───────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        ep_dir, t = self._index[idx]
        ep = self._load_episode(ep_dir)

        T = ep["actions"].shape[0]

        # ── State ─────────────────────────────────────────────────────────
        state = ep["qpos"][t].copy()                   # [actual_state_dim] — physical space
        actual_dim = state.shape[0]

        # Quantile-normalize the actual_dim state BEFORE padding.
        # Mirrors the inference transform ordering:
        #   Normalize(actual_dim) → TokenizePrompt(actual_dim) → PadStatesAndActions(action_dim)
        # z-score here would cause a distribution mismatch at inference because
        # softvla_model uses Normalize(use_quantiles=True) for PI05.
        state_norm = state.astype(np.float32)
        if self.norm_stats is not None and "state" in self.norm_stats:
            ns = self.norm_stats["state"]
            q01 = np.asarray(ns.q01, dtype=np.float32)[:actual_dim]
            q99 = np.asarray(ns.q99, dtype=np.float32)[:actual_dim]
            state_norm = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            state_norm = np.clip(state_norm, -1.0, 1.0)

        # ── Action chunk [action_horizon, action_dim] ──────────────────────
        end = min(t + self.action_horizon, T)
        chunk_len = end - t
        raw_action = ep["actions"][t:end]  # [chunk_len, actual_action_dim] — xyz is absolute on disk

        action_chunk = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        # Fill valid timesteps (zero-pad dim if needed)
        fill_dim = min(raw_action.shape[1], self.action_dim)
        action_chunk[:chunk_len, :fill_dim] = raw_action[:, :fill_dim]
        # Repeat last action for temporal padding
        if chunk_len < self.action_horizon:
            action_chunk[chunk_len:] = action_chunk[chunk_len - 1]

        # Apply chunk-level xyz delta: subtract current state xyz from ALL steps.
        # actions on disk are absolute; model expects delta relative to state_t.
        # Matches DeltaActions in RoboTwinEEDataConfig (openpi/transforms.py).
        action_chunk[:, :3] -= state[:3]

        # Quantile-normalize action chunk (matches inference Normalize transform).
        # norm_stats key is "actions" plural — matches upstream openpi convention.
        # Zero-padded dims (actual_action_dim:action_dim) use q01=-1, q99=1
        # so their zero values map to 0.0 after normalization.
        if self.norm_stats is not None and "actions" in self.norm_stats:
            na = self.norm_stats["actions"]
            q01_a = np.asarray(na.q01, dtype=np.float32)
            q99_a = np.asarray(na.q99, dtype=np.float32)
            if len(q01_a) < self.action_dim:
                q01_a = np.pad(q01_a, (0, self.action_dim - len(q01_a)), constant_values=-1.0)
                q99_a = np.pad(q99_a, (0, self.action_dim - len(q99_a)), constant_values=1.0)
            action_chunk = (action_chunk - q01_a) / (q99_a - q01_a + 1e-6) * 2.0 - 1.0

        # ── Instruction (random pick among seen instructions) ─────────────
        instructions = ep["instructions"]
        instr = instructions[np.random.randint(len(instructions))] if instructions else ""

        # ── Tokenize with actual_dim state (Pi05 discrete_state_input=True) ────
        # state_norm is actual_dim (NOT yet padded): mirrors inference where
        # TokenizePrompt runs BEFORE PadStatesAndActions in model_transforms.inputs.
        # Passing the full 32D padded state would produce 32 state tokens at
        # training time vs. 10 tokens at inference — a prompt length mismatch.
        tokens, token_mask = self.tokenizer.tokenize(instr, state=state_norm)

        # ── Images ────────────────────────────────────────────────────────
        cam_high        = self._decode_image(ep["cam_high"][t])
        cam_right_wrist = self._decode_image(ep["cam_right_wrist"][t])
        cam_left_wrist  = self._decode_image(ep["cam_left_wrist"][t])

        return {
            # Images: [H, W, 3] float32  (DataLoader stacks to [B, H, W, 3])
            "cam_high":               cam_high,
            "cam_right_wrist":        cam_right_wrist,
            "cam_left_wrist":         cam_left_wrist,
            # State: [action_dim] float32 (quantile-normalised, then zero-padded to action_dim)
            # Padding mirrors PadStatesAndActions which runs after TokenizePrompt in inference.
            "state":                  np.pad(state_norm, (0, self.action_dim - actual_dim)).astype(np.float32),
            # Tokens: [max_token_len]
            "tokenized_prompt":       tokens.astype(np.int64),
            "tokenized_prompt_mask":  token_mask.astype(bool),
            # Action chunk: [action_horizon, action_dim] float32 (normalized)
            "action":                 action_chunk,
            # Scalar
            "domain_id":              np.int64(self.domain_id),
        }
