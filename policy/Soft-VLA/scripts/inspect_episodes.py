"""
Inspect processed HDF5 episode data.

For each episode directory under a given root, this script:
  - Extracts raw action arrays and saves them as .npy + a human-readable .csv
  - Writes a JSON summary containing action shape, action_dim, and per-step stats
  - Renders images from every camera into an mp4 video (cameras tiled side-by-side)

Usage
-----
    python inspect_episodes.py [ROOT_DIR] [OUTPUT_DIR]

    ROOT_DIR   : directory that contains episode_* sub-dirs  (default: ../processed_data from this script)
    OUTPUT_DIR : where to store outputs  (default: ROOT_DIR/inspection_output)

Each episode produces:
    <OUTPUT_DIR>/<episode_name>/action.npy
    <OUTPUT_DIR>/<episode_name>/action.csv
    <OUTPUT_DIR>/<episode_name>/action_summary.json
    <OUTPUT_DIR>/<episode_name>/video.mp4
"""

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = (SCRIPT_DIR.parent / "processed_data").resolve()


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
VIDEO_FPS = 10


def decode_jpeg(raw: bytes) -> np.ndarray:
    """Decode JPEG bytes → uint8 RGB ndarray (H, W, 3)."""
    img = Image.open(io.BytesIO(raw))
    return np.array(img.convert("RGB"))


def tile_images(imgs: list[np.ndarray]) -> np.ndarray:
    """Horizontally concatenate a list of images, resizing to the same height."""
    if not imgs:
        raise ValueError("No images to tile")
    h = imgs[0].shape[0]
    resized = []
    for im in imgs:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h))
        resized.append(im)
    return np.concatenate(resized, axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# per-episode processing
# ──────────────────────────────────────────────────────────────────────────────

def process_episode(hdf5_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        action: np.ndarray = f["action"][:]                        # (T, D)
        left_dim = f["observations/left_arm_dim"][:] if "observations/left_arm_dim" in f else None
        right_dim = f["observations/right_arm_dim"][:] if "observations/right_arm_dim" in f else None

        # ── action output ────────────────────────────────────────────────────
        np.save(out_dir / "action.npy", action)

        # CSV: header row describes each column
        action_dim = int(action.shape[1])
        col_names = [f"action_{i}" for i in range(action_dim)]
        csv_lines = [",".join(col_names)]
        for row in action:
            csv_lines.append(",".join(f"{v:.6f}" for v in row))
        (out_dir / "action.csv").write_text("\n".join(csv_lines))

        # JSON summary
        def _dim_value(dim_arr):
            if dim_arr is None:
                return None
            uniq = np.unique(dim_arr)
            return int(uniq[0]) if len(uniq) == 1 else dim_arr.tolist()

        summary = {
            "episode": hdf5_path.stem,
            "num_steps": int(action.shape[0]),
            "action_dim": action_dim,
            "left_arm_dim": _dim_value(left_dim),
            "right_arm_dim": _dim_value(right_dim),
            "action_format": f.attrs.get("action_format", None),
            "action_layout": f.attrs.get("action_layout", None),
            "action_stats": {
                "min":  action.min(axis=0).tolist(),
                "max":  action.max(axis=0).tolist(),
                "mean": action.mean(axis=0).tolist(),
                "std":  action.std(axis=0).tolist(),
            },
        }
        (out_dir / "action_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # ── video output ─────────────────────────────────────────────────────
        T = action.shape[0]
        writer = None

        for t in range(T):
            frames = []
            for cam_key in CAMERA_KEYS:
                key = f"observations/images/{cam_key}"
                if key in f:
                    raw = bytes(f[key][t])
                    rgb = decode_jpeg(raw)
                    frames.append(rgb)

            if not frames:
                continue

            tiled_rgb = tile_images(frames)          # (H, W*num_cams, 3)
            tiled_bgr = cv2.cvtColor(tiled_rgb, cv2.COLOR_RGB2BGR)

            if writer is None:
                h, w = tiled_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_dir / "video.mp4"), fourcc, VIDEO_FPS, (w, h)
                )

            writer.write(tiled_bgr)

        if writer is not None:
            writer.release()

    print(f"  [OK] {hdf5_path.parent.name} → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=str(DEFAULT_ROOT_DIR),
        help=f"Root directory containing episode_* subdirectories (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument("output_dir", nargs="?", default=None, help="Output base directory (default: <root_dir>/inspection_output)")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    out_base = Path(args.output_dir).resolve() if args.output_dir else root / "inspection_output"

    episode_dirs = sorted(root.glob("episode_*"))
    if not episode_dirs:
        print(f"[ERROR] No episode_* directories found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Root    : {root}")
    print(f"Output  : {out_base}")
    print(f"Episodes: {len(episode_dirs)}")
    print()

    for ep_dir in episode_dirs:
        hdf5_files = list(ep_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"  [SKIP] {ep_dir.name} — no .hdf5 file found")
            continue
        if len(hdf5_files) > 1:
            print(f"  [WARN] {ep_dir.name} — multiple .hdf5 files, using first: {hdf5_files[0].name}")

        process_episode(hdf5_files[0], out_base / ep_dir.name)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
