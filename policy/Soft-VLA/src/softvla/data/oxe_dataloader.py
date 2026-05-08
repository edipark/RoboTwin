"""OXE streaming data loader for Soft-VLA Phase 1 & 2 training.

Mirrors the pattern from openpi/training/droid_rlds_dataset.py:
  - Loads multiple RLDS datasets via TensorFlow Datasets + dlimp.
  - Mixes them with equal per-dataset weights.
  - Maps each step to the observation format expected by SoftVLAPytorch.
  - Tokenizes language instructions with the PaliGemma processor.
  - Yields numpy batches: (obs_dict, actions_np [B,H,action_dim], domain_ids_np [B]).

TensorFlow is imported lazily and GPU devices are hidden from TF at import time
so it does not contend with PyTorch for GPU memory.

Observation dict keys (mirror fake_observation in train_softvla.py):
    images:
        "base_0_rgb"         : [B, 224, 224, 3]  float32 in [-1, 1]
        "left_wrist_0_rgb"   : [B, 224, 224, 3]  float32 in [-1, 1]
    image_masks:
        "base_0_rgb"         : [B]  bool
        "left_wrist_0_rgb"   : [B]  bool
    state                    : [B, action_dim]  float32 (zero-padded)
    tokenized_prompt         : [B, max_token_len]  int32
    tokenized_prompt_mask    : [B, max_token_len]  bool
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
import logging
import os
from typing import Any

import numpy as np

from softvla.data.oxe_config import DATASET_REGISTRY
from softvla.data.oxe_config import DatasetConfig
from softvla.data.rotation_utils import action_7d_to_10d
from softvla.data.rotation_utils import pad_to_dim

# Image size expected by PaliGemma SigLIP ViT.
_IMG_SIZE = 224
# Language prompt max length (must match SoftVLAConfig.max_token_len).
_DEFAULT_MAX_TOKEN_LEN = 200
# Hugging Face model id for PaliGemma tokenizer.
_PALIGEMMA_ID = "google/paligemma-3b-pt"
# Fixed action/state dims for uniform TF batching across heterogeneous datasets.
# action_7d_to_10d() in rotation_utils assumes exactly 7D actions.
# _MAX_STATE_DIM must be <= SoftVLAConfig.action_dim (default 32).
_MAX_ACT_DIM: int = 7
_MAX_STATE_DIM: int = 32


# ── Tokenizer (singleton) ────────────────────────────────────────────────────

class _SentencePieceProcessor:
    """PaliGemma tokenizer backed by a local SentencePiece .model file.

    Uses the same SPM model as openpi (downloaded from
    gs://big_vision/paligemma_tokenizer.model).  BOS token is prepended to
    match the HuggingFace AutoProcessor behaviour.
    """

    # GCS path used by openpi; anonymous access is allowed.
    _GCS_PATH = "gs://big_vision/paligemma_tokenizer.model"
    # Standard local cache written by openpi's download helper.
    _LOCAL_CACHE = os.path.expanduser(
        "~/.cache/openpi/big_vision/paligemma_tokenizer.model"
    )
    # PaliGemma BOS id.
    _BOS_ID: int = 2

    class _Tokenizer:
        def __init__(self, sp):
            self._sp = sp

        def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
            import sentencepiece as spm  # noqa: F401 (ensure available)
            bos = _SentencePieceProcessor._BOS_ID
            ids_out = np.zeros((len(texts), max_length), dtype=np.int32)
            attn_out = np.zeros((len(texts), max_length), dtype=np.int64)
            for i, text in enumerate(texts):
                toks = [bos, *self._sp.encode(text, out_type=int)]
                if truncation:
                    toks = toks[:max_length]
                length = min(len(toks), max_length)
                ids_out[i, :length] = toks[:length]
                attn_out[i, :length] = 1

            class _Enc:
                def __getitem__(self, key):
                    return ids_out if key == "input_ids" else attn_out

            return _Enc()

    @classmethod
    def _load(cls):
        """Load SPM model from cache or GCS."""
        import sentencepiece as spm

        model_path = cls._LOCAL_CACHE
        if not os.path.exists(model_path):
            try:
                import openpi.shared.download as _dl
                dl_path = _dl.maybe_download(cls._GCS_PATH, gs={"token": "anon"})
                model_path = str(dl_path)
            except Exception as exc:
                raise FileNotFoundError(
                    f"SentencePiece model not found at {cls._LOCAL_CACHE} "
                    f"and GCS download failed: {exc}"
                ) from exc

        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return sp

    def __init__(self):
        sp = self._load()
        self.tokenizer = self._Tokenizer(sp)


class _DummyProcessor:
    """Fallback tokeniser that returns zero tensors.

    Used when the PaliGemma processor is unavailable (e.g. no HuggingFace
    credentials / no network).  Smoke tests and pipeline validation can still
    run; the zero tokens produce meaningless language conditioning but the
    numerical shapes are correct.
    """

    class _Tokenizer:
        def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
            n = len(texts)
            ids = np.zeros((n, max_length), dtype=np.int32)
            attn = np.zeros((n, max_length), dtype=np.int64)
            class _Enc:
                def __getitem__(self, key):
                    return ids if key == "input_ids" else attn
            return _Enc()

    def __init__(self):
        self.tokenizer = self._Tokenizer()


@lru_cache(maxsize=1)
def _get_processor(model_id: str, local_dir: str | None = None):
    """Load PaliGemma processor once and cache it.

    Falls back to a zero-tensor dummy processor when the real one cannot be
    loaded (e.g. missing HuggingFace credentials or no network access).

    Args:
        model_id: HuggingFace model identifier.
        local_dir: Optional path to a cached local copy (e.g. norm_stats_dir/tokenizer/).

    Returns:
        AutoProcessor instance (or _DummyProcessor fallback).
    """
    from transformers import AutoProcessor

    candidate = None
    if local_dir is not None:
        tok_path = os.path.join(local_dir, "tokenizer")
        if os.path.isdir(tok_path):
            candidate = tok_path

    # If a local candidate dir is provided, try AutoProcessor against it first.
    # Otherwise skip the HF network round-trip when no token is available:
    # attempt order is (1) local dir → (2) SentencePiece cache → (3) HF network → (4) dummy.
    hf_token = (
        os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    _hf_token_file = os.path.expanduser("~/.cache/huggingface/token")
    if not hf_token and os.path.isfile(_hf_token_file):
        with open(_hf_token_file) as _f:
            hf_token = _f.read().strip() or None

    if candidate is not None:
        # Local directory specified — try AutoProcessor directly (no network).
        logging.info("[OXE] Loading PaliGemma processor from local dir '%s' …", candidate)
        try:
            return AutoProcessor.from_pretrained(candidate, use_fast=True, local_files_only=True)
        except Exception as exc:
            logging.warning("[OXE] Local AutoProcessor failed ('%s'): %s.", candidate, exc)

    # Try SentencePiece (cached GCS model — no HF account needed).
    try:
        proc = _SentencePieceProcessor()
        logging.info("[OXE] PaliGemma SentencePiece tokenizer loaded (language conditioning active).")
        return proc
    except Exception as exc_spm:
        logging.debug("[OXE] SentencePiece load failed: %s.", exc_spm)

    # Fall back to HF network only when a token is present.
    if hf_token:
        logging.info("[OXE] Trying HuggingFace AutoProcessor for '%s' …", model_id)
        try:
            return AutoProcessor.from_pretrained(model_id, use_fast=True, token=hf_token)
        except Exception as exc_hf:
            logging.warning("[OXE] HuggingFace AutoProcessor failed: %s.", exc_hf)

    logging.warning(
        "[OXE] All tokenizer sources failed. "
        "Using dummy zero-token fallback — language conditioning will be inactive."
    )
    return _DummyProcessor()


def _tokenize(texts: list[str], processor, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a list of strings to padded numpy arrays.

    Returns:
        token_ids:  [N, max_len]  int32
        token_mask: [N, max_len]  bool
    """
    enc = processor.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(bool)


# ── Per-step TF map functions ────────────────────────────────────────────────

def _make_restructure_fn(cfg: DatasetConfig):
    """Return a traj_map function that normalises observation/action keys.

    Follows openpi's droid_rlds_dataset.py pattern: direct nested-dict access
    at Python (trace) time.  Decorated with do_not_convert so autograph never
    modifies the dict-traversal logic.

    All per-dataset key resolution happens in the closure (Python time) so the
    inner function only sees concrete strings / Python bools.
    """
    import tensorflow as tf

    # ── Closure constants (resolved once, not traced by autograph) ───────────
    action_keys     = cfg.action_keys          # tuple[str, ...]
    action_max_dims = cfg.action_key_max_dim   # tuple[int|None, ...]
    has_wrist       = cfg.wrist_image_key is not None
    domain_id       = cfg.domain_id            # Python int

    def _deep_get(d: dict, slash_key: str):
        """Greedy nested-dict traversal supporting partial nesting.

        Handles both fully-nested schemas (traj["a"]["b"]["c"]) and
        partially-nested ones where intermediate nodes keep slash keys
        (e.g. bc_z: traj["action"]["future/xyz_residual"]).
        """
        parts = slash_key.split("/")
        cur = d
        i = 0
        while i < len(parts):
            # Try longest possible remaining key first (greedy)
            found = False
            for j in range(len(parts), i, -1):
                candidate = "/".join(parts[i:j])
                if isinstance(cur, dict) and candidate in cur:
                    cur = cur[candidate]
                    i = j
                    found = True
                    break
            if not found:
                available = list(cur.keys()) if isinstance(cur, dict) else type(cur)
                raise KeyError(
                    f"Cannot find '{parts[i]}' in {available} "
                    f"(full path: '{slash_key}', resolved up to index {i})"
                )
        return cur

    @tf.autograph.experimental.do_not_convert
    def restructure(traj):
        # 1. Primary image (encoded bytes, decoded later per frame)
        base_img_enc = _deep_get(traj, cfg.primary_image_key)

        # 2. Action: gather each key, ensure 2D [T, D], optional dim-slice, concat
        action_parts = []
        for key, max_d in zip(action_keys, action_max_dims, strict=True):
            t = _deep_get(traj, key)
            t = tf.cast(t, tf.float32)
            if len(t.shape) == 1:          # [T] scalar-per-step → [T, 1]
                t = tf.expand_dims(t, -1)
            if max_d is not None:
                t = t[:, :max_d]
            action_parts.append(t)
        action = tf.concat(action_parts, axis=-1) if len(action_parts) > 1 else action_parts[0]

        # 3. State
        state = tf.cast(_deep_get(traj, cfg.state_key), tf.float32)

        # Normalize action and state to fixed dims for uniform batching.
        # Clip first (handles datasets wider than target), then zero-pad.
        action = action[:, :_MAX_ACT_DIM]
        act_pad = _MAX_ACT_DIM - tf.shape(action)[1]
        action = tf.pad(action, [[0, 0], [0, act_pad]])
        action = tf.ensure_shape(action, [None, _MAX_ACT_DIM])

        state = state[:, :_MAX_STATE_DIM]
        st_pad = _MAX_STATE_DIM - tf.shape(state)[1]
        state = tf.pad(state, [[0, 0], [0, st_pad]])
        state = tf.ensure_shape(state, [None, _MAX_STATE_DIM])

        # 4. Language instruction
        language = _deep_get(traj, cfg.language_key)

        # 5. Shape reference: is_first is ALWAYS a plain [T] bool tensor in RLDS.
        #    Use it instead of traj["action"] to avoid KeyErrors when action is a dict.
        ref_shape = tf.shape(traj["is_first"])   # 1D int tensor containing T

        # 6. Wrist image — uniform schema across all datasets; placeholder when absent
        if has_wrist:
            wrist_img_enc = _deep_get(traj, cfg.wrist_image_key)
        else:
            wrist_img_enc = tf.broadcast_to(tf.constant(b""), ref_shape)

        return {
            "base_img_enc":  base_img_enc,
            "wrist_img_enc": wrist_img_enc,
            "has_wrist":     tf.broadcast_to(tf.constant(has_wrist),                 ref_shape),
            "action":        action,
            "state":         state,
            "language":      language,
            "domain_id":     tf.broadcast_to(tf.constant(domain_id, dtype=tf.int32), ref_shape),
        }

    return restructure


def _make_chunk_fn(action_horizon: int):
    """Return a traj_map that chunks actions into sequences of *action_horizon*."""
    import tensorflow as tf

    def chunk_actions(traj):
        traj_len = tf.shape(traj["action"])[0]
        indices = (
            tf.broadcast_to(tf.range(action_horizon)[None], [traj_len, action_horizon])
            + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, action_horizon])
        )
        indices = tf.minimum(indices, traj_len - 1)
        traj["action"] = tf.gather(traj["action"], indices)   # [T, H, 7]
        return traj

    return chunk_actions


def _make_tfds_builder(cfg: DatasetConfig, oxe_data_dir: str):
    """Create a TFDS builder that prefers the prepared local dataset directory.

    `tfds.builder(..., version=...)` can still consult the registered builder
    code path, which may drift from the local prepared dataset schema. For OXE
    pre-training we want the schema stored alongside the prepared dataset when
    it is available locally.
    """
    import tensorflow_datasets as tfds

    ds_dir = os.path.join(oxe_data_dir, cfg.tfds_name, cfg.version)
    if os.path.isdir(ds_dir):
        return tfds.builder_from_directory(ds_dir)
    return tfds.builder(cfg.tfds_name, data_dir=oxe_data_dir, version=cfg.version)


def _decode_and_resize(frame: dict[str, Any]) -> dict[str, Any]:
    """Decode encoded images and resize to _IMG_SIZE x _IMG_SIZE."""
    import tensorflow as tf

    def _decode(enc):
        img = tf.io.decode_image(enc, expand_animations=False, dtype=tf.uint8)
        img = tf.image.resize(img, [_IMG_SIZE, _IMG_SIZE], method="bilinear")
        return tf.cast(img, tf.uint8)

    frame["base_img"] = _decode(frame["base_img_enc"])
    # Use tf.cond (not Python if) so both branches define wrist_img and schemas stay consistent.
    # The false branch runs for datasets without a wrist camera (has_wrist=False).
    frame["wrist_img"] = tf.cond(
        frame["has_wrist"],
        lambda: _decode(frame["wrist_img_enc"]),
        lambda: tf.zeros([_IMG_SIZE, _IMG_SIZE, 3], dtype=tf.uint8),
    )
    return frame


# ── Main public API ───────────────────────────────────────────────────────────

def create_oxe_data_loader(
    oxe_data_dir: str,
    action_horizon: int,
    action_dim: int,
    batch_size: int,
    norm_stats_dir: str | None,
    shuffle: bool = True,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
    num_parallel_calls: int = 4,    # Fixed (not AUTOTUNE): limits in-flight decoded frames per dataset
    num_parallel_reads: int = 4,    # Fixed (not AUTOTUNE): limits concurrent shard reads per dataset
    max_token_len: int = _DEFAULT_MAX_TOKEN_LEN,
    num_prefetch_batches: int = 4,
) -> "OXEDataLoader":
    """Construct the OXE data loader.

    Args:
        oxe_data_dir:      Root directory containing all RLDS dataset folders.
        action_horizon:    Number of future actions to predict per step (H).
        action_dim:        Total action dimension in the model (default 32).
        batch_size:        Per-process batch size.
        norm_stats_dir:    Path to normalisation statistics (used to locate
                           a cached tokenizer if present). May be None.
        shuffle:           Whether to shuffle trajectories.
        shuffle_buffer:    Shuffle buffer size (number of frames).
        seed:              Random seed for TF dataset shuffling.
        num_parallel_calls: TF dataset map parallelism (-1 = AUTOTUNE).
        num_parallel_reads: TF dataset read parallelism (-1 = AUTOTUNE).
        max_token_len:     Maximum tokenised prompt length.

    Returns:
        An :class:`OXEDataLoader` that yields
        ``(obs_dict, actions_np [B,H,action_dim], domain_ids_np [B])`` tuples.
    """
    return OXEDataLoader(
        oxe_data_dir=oxe_data_dir,
        action_horizon=action_horizon,
        action_dim=action_dim,
        batch_size=batch_size,
        norm_stats_dir=norm_stats_dir,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        num_parallel_calls=num_parallel_calls,
        num_parallel_reads=num_parallel_reads,
        max_token_len=max_token_len,
        num_prefetch_batches=num_prefetch_batches,
    )


class OXEDataLoader:
    """Streaming data loader over multiple OXE RLDS datasets.

    Yields numpy batches of the form::

        obs_dict : {
            "images"       : {"base_0_rgb": [B,224,224,3], "left_wrist_0_rgb": [B,224,224,3]},
            "image_masks"  : {"base_0_rgb": [B], "left_wrist_0_rgb": [B]},
            "state"        : [B, action_dim],
            "tokenized_prompt"      : [B, max_token_len],
            "tokenized_prompt_mask" : [B, max_token_len],
        }
        actions_np    : np.ndarray  [B, action_horizon, action_dim]  float32
        domain_ids_np : np.ndarray  [B]  int32
    """

    def __init__(
        self,
        oxe_data_dir: str,
        action_horizon: int,
        action_dim: int,
        batch_size: int,
        norm_stats_dir: str | None,
        shuffle: bool,
        shuffle_buffer: int,
        seed: int,
        num_parallel_calls: int,
        num_parallel_reads: int,
        max_token_len: int,
        num_prefetch_batches: int = 4,
    ):
        # Prevent TF from grabbing GPU memory.
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")

        import dlimp as dl

        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._max_token_len = max_token_len
        self._norm_stats_dir = norm_stats_dir

        logging.info("[OXE] Building data pipeline for %d datasets …", len(DATASET_REGISTRY))

        def _build_single(cfg: DatasetConfig):
            """Load, restructure, and chunk one RLDS dataset."""
            ds_dir = os.path.join(oxe_data_dir, cfg.tfds_name)
            if not os.path.isdir(ds_dir):
                raise FileNotFoundError(
                    f"Dataset directory not found: {ds_dir}\n"
                    f"Make sure oxe_data_dir='{oxe_data_dir}' contains '{cfg.tfds_name}/'"
                )

            builder = _make_tfds_builder(cfg, oxe_data_dir)
            dataset = dl.DLataset.from_rlds(
                builder,
                split="train",
                shuffle=shuffle,
                num_parallel_reads=num_parallel_reads,
            )
            dataset = dataset.repeat()

            # Restructure trajectories.
            restructure = _make_restructure_fn(cfg)
            dataset = dataset.traj_map(restructure, num_parallel_calls)

            # Chunk actions → [T, H, 7]
            dataset = dataset.traj_map(_make_chunk_fn(action_horizon), num_parallel_calls)

            # Flatten to per-step frames.
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # Decode & resize images per frame.
            return dataset.frame_map(_decode_and_resize, num_parallel_calls)

        all_datasets = [_build_single(cfg) for cfg in DATASET_REGISTRY]
        weights = [cfg.weight for cfg in DATASET_REGISTRY]

        combined = dl.DLataset.sample_from_datasets(all_datasets, weights=weights, seed=seed)
        if shuffle:
            combined = combined.shuffle(shuffle_buffer, seed=seed)
        combined = combined.batch(batch_size)
        combined = combined.with_ram_budget(4)  # 4 GB per process x 8 procs = 32 GB total; tighter hint reduces TF autotune buffering

        self._dataset = combined
        self._batch_size = batch_size
        self._num_prefetch_batches = num_prefetch_batches

    # ── Post-processing (numpy → structured obs_dict) ────────────────────────

    def _process_batch(self, raw: dict) -> tuple[dict, np.ndarray, np.ndarray]:
        """Convert a raw numpy batch into the structured format."""
        batch_size = raw["action"].shape[0]
        processor = _get_processor(_PALIGEMMA_ID, self._norm_stats_dir)

        # ── Images ──────────────────────────────
        # raw["base_img"]: [B, 224, 224, 3]  uint8
        base_img = raw["base_img"].astype(np.float32) / 127.5 - 1.0  # → [-1, 1]

        has_wrist = bool(raw.get("has_wrist", np.array([False]))[0])
        if has_wrist and "wrist_img" in raw:
            wrist_img = raw["wrist_img"].astype(np.float32) / 127.5 - 1.0
        else:
            wrist_img = np.zeros_like(base_img)

        images = {
            "base_0_rgb": base_img,
            "left_wrist_0_rgb": wrist_img,
            # right_wrist_0_rgb: not present in OXE; omitted here, added in batch_to_torch
        }
        image_masks = {
            "base_0_rgb": np.ones(batch_size, dtype=bool),
            "left_wrist_0_rgb": np.full(batch_size, has_wrist, dtype=bool),
        }

        # ── State ────────────────────────────────
        state_raw = raw["state"].astype(np.float32)  # [B, state_dim]
        state = pad_to_dim(state_raw, self._action_dim)  # [B, action_dim]

        # ── Language ────────────────────────────
        lang = raw.get("language")
        if lang is None or (isinstance(lang, np.ndarray) and lang.size == 0):
            texts = [""] * batch_size
        elif isinstance(lang, np.ndarray) and lang.ndim == 0:
            texts = [lang.item().decode("utf-8") if isinstance(lang.item(), bytes) else str(lang.item())] * batch_size
        elif isinstance(lang, np.ndarray) and lang.ndim >= 1:
            texts = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in lang.reshape(-1)[:batch_size]
            ]
        else:
            texts = [str(lang)] * batch_size

        token_ids, token_mask = _tokenize(texts, processor, self._max_token_len)

        # ── Actions ──────────────────────────────
        # raw["action"]: [B, H, 7]  → 10D → 32D
        actions_7d = raw["action"].astype(np.float32)   # [B, H, 7]
        actions_10d = action_7d_to_10d(actions_7d)      # [B, H, 10]
        actions = pad_to_dim(actions_10d, self._action_dim)  # [B, H, 32]

        # ── Domain IDs ───────────────────────────
        domain_ids = raw["domain_id"].astype(np.int32).reshape(batch_size)

        obs_dict = {
            "images": images,
            "image_masks": image_masks,
            "state": state,
            "tokenized_prompt": token_ids,
            "tokenized_prompt_mask": token_mask,
        }
        return obs_dict, actions, domain_ids

    def __iter__(self) -> Iterator[tuple[dict, np.ndarray, np.ndarray]]:
        """Yield processed batches, prefetching `num_prefetch_batches` ahead in a background thread."""
        import queue
        import threading

        q: queue.Queue = queue.Queue(maxsize=self._num_prefetch_batches)
        stop_sentinel = object()

        def _producer():
            try:
                for raw_batch in self._dataset.as_numpy_iterator():
                    q.put(self._process_batch(raw_batch))
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(stop_sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is stop_sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
