"""Inspect processed HDF5 episode data — action analytics.

For each episode, this script:
  - Scans ALL numeric datasets in the HDF5 and reports their shapes
  - Extracts action arrays → .npy + .csv
  - Computes detailed per-dimension statistics (min/max/mean/std/median/IQR/
    skewness/kurtosis/percentiles) → action_summary.json
  - Generates per-episode plots:
      plot_timeseries.png   — action dims over time (grouped by semantic role)
      plot_histograms.png   — per-dim distribution histograms
      plot_boxplots.png     — side-by-side boxplots across all dims
      plot_correlation.png  — dim-to-dim Pearson correlation heatmap
  - Generates a cross-episode aggregate plot → <OUTPUT_DIR>/aggregate_plot.png

Usage
-----
    python inspect_episodes.py [ROOT_DIR] [OUTPUT_DIR] [--task TASK ...] [--no-plots]

    ROOT_DIR   : directory containing episode_* or task sub-dirs
                 (default: ../processed_data relative to this script)
    OUTPUT_DIR : where to store outputs  (default: ROOT_DIR/inspection_output)
    --task     : restrict to specific task sub-directories
    --no-plots : skip matplotlib figure generation (faster, JSON/CSV only)
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = (SCRIPT_DIR.parent / "processed_data").resolve()

# ──────────────────────────────────────────────────────────────────────────────
# Semantic dim labels for known action layouts
# ──────────────────────────────────────────────────────────────────────────────

_LAYOUT_LABELS: dict[str, list[str]] = {
    "right_only_10d": [
        "rx (Δ)", "ry (Δ)", "rz (Δ)",            # xyz delta
        "r6d_0", "r6d_1", "r6d_2",               # rot6d row-0
        "r6d_3", "r6d_4", "r6d_5",               # rot6d row-1
        "r_grip",                                  # gripper
    ],
    "dual_arm_14d": [
        "lx", "ly", "lz", "lqw", "lqx", "lqy", "lqz",
        "rx", "ry", "rz", "rqw", "rqx", "rqy", "rqz",
    ],
}

# Colour groups for time-series (index ranges → colour)
_LAYOUT_GROUPS: dict[str, list[tuple[range, str, str]]] = {
    "right_only_10d": [
        (range(0, 3),  "#e06c75", "xyz (delta)"),
        (range(3, 9),  "#61afef", "rot6d"),
        (range(9, 10), "#98c379", "gripper"),
    ],
}


def _dim_labels(action_dim: int, layout: str | None) -> list[str]:
    """Return human-readable labels for each action dim."""
    if layout and layout in _LAYOUT_LABELS:
        labels = _LAYOUT_LABELS[layout]
        if len(labels) >= action_dim:
            return labels[:action_dim]
    return [f"dim_{i}" for i in range(action_dim)]


def _dim_groups(action_dim: int, layout: str | None):
    """Return colour groups or a single fallback group."""
    if layout and layout in _LAYOUT_GROUPS:
        return _LAYOUT_GROUPS[layout]
    # fallback: all dims the same colour
    return [(range(0, action_dim), "#abb2bf", "action")]


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 dataset scanner
# ──────────────────────────────────────────────────────────────────────────────

def scan_hdf5(f: h5py.File) -> dict[str, dict]:
    """Walk entire HDF5 tree; return {path: {shape, dtype}} for numeric datasets."""
    result: dict[str, dict] = {}

    def _visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if not np.issubdtype(obj.dtype, np.number):
            return
        result[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

    f.visititems(_visit)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(arr: np.ndarray) -> dict:
    """Per-dimension detailed statistics for a 2-D array (T, D)."""
    if arr.ndim == 1:
        arr = arr[:, None]
    q5, q25, q50, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95], axis=0)
    skew = scipy_stats.skew(arr, axis=0).tolist()
    kurt = scipy_stats.kurtosis(arr, axis=0).tolist()
    return {
        "min":    arr.min(axis=0).tolist(),
        "max":    arr.max(axis=0).tolist(),
        "range":  (arr.max(axis=0) - arr.min(axis=0)).tolist(),
        "mean":   arr.mean(axis=0).tolist(),
        "std":    arr.std(axis=0).tolist(),
        "median": q50.tolist(),
        "q5":     q5.tolist(),
        "q25":    q25.tolist(),
        "q75":    q75.tolist(),
        "q95":    q95.tolist(),
        "iqr":    (q75 - q25).tolist(),
        "skew":   skew,
        "kurtosis": kurt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG = "#282c34"
AXES_BG = "#21252b"
GRID_COLOR = "#3e4451"
TEXT_COLOR = "#abb2bf"

def _apply_dark_style(fig, axes_iter):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_iter:
        ax.set_facecolor(AXES_BG)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5)


def plot_timeseries(action: np.ndarray, labels: list[str], groups, out_path: Path, title: str) -> None:
    T, D = action.shape
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(max(10, T // 5), 5))
    for idx_range, color, group_label in groups:
        first = True
        for i in idx_range:
            if i >= D:
                break
            ax.plot(t, action[:, i], color=color, linewidth=0.9, alpha=0.85,
                    label=f"{group_label}: {labels[i]}" if first else labels[i])
            first = False

    ax.set_xlabel("timestep")
    ax.set_ylabel("value")
    ax.set_title(title)
    # Legend: one entry per group
    handles, leg_labels = ax.get_legend_handles_labels()
    # Keep only first entry of each group
    seen_groups = set()
    filtered = []
    for h, l in zip(handles, leg_labels):
        grp = l.split(":")[0].strip() if ":" in l else None
        if grp and grp not in seen_groups:
            seen_groups.add(grp)
            filtered.append((h, grp))
    if filtered:
        ax.legend(*zip(*filtered), fontsize=7, loc="upper right",
                  facecolor=AXES_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(action: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    D = action.shape[1]
    ncols = min(5, D)
    nrows = (D + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    axes_flat = np.array(axes).flatten() if D > 1 else [axes]

    for i in range(D):
        ax = axes_flat[i]
        data = action[:, i]
        ax.hist(data, bins=40, color="#61afef", edgecolor="none", alpha=0.85)
        ax.axvline(data.mean(), color="#e06c75", linewidth=1.2, linestyle="--", label=f"μ={data.mean():.3f}")
        ax.axvline(np.median(data), color="#98c379", linewidth=1.0, linestyle=":", label=f"med={np.median(data):.3f}")
        ax.set_title(labels[i], fontsize=8)
        ax.set_xlabel("value", fontsize=7)
        ax.legend(fontsize=6, facecolor=AXES_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Hide unused subplots
    for j in range(D, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, color=TEXT_COLOR, fontsize=10)
    _apply_dark_style(fig, axes_flat[:D])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(action: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    D = action.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, D * 0.9), 5))

    bp = ax.boxplot(
        [action[:, i] for i in range(D)],
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="#e06c75", linewidth=1.5),
        whiskerprops=dict(color=TEXT_COLOR),
        capprops=dict(color=TEXT_COLOR),
        flierprops=dict(marker="o", markerfacecolor="#e5c07b", markersize=2, alpha=0.4, linestyle="none"),
    )
    colors = plt.cm.Set2(np.linspace(0, 1, D))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(action: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    if action.shape[1] < 2:
        return
    corr = np.corrcoef(action.T)           # (D, D)
    D = corr.shape[0]

    fig, ax = plt.subplots(figsize=(max(5, D * 0.7), max(4, D * 0.65)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)

    # Annotate cells
    for i in range(D):
        for j in range(D):
            color = "white" if abs(corr[i, j]) > 0.5 else TEXT_COLOR
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6, color=color)

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(all_actions: list[np.ndarray], labels: list[str], out_path: Path, title: str) -> None:
    """Cross-episode mean ± std band per dim."""
    if not all_actions:
        return
    # Pad/trim to equal length (use min length for fair comparison)
    min_T = min(a.shape[0] for a in all_actions)
    stacked = np.stack([a[:min_T] for a in all_actions], axis=0)   # (N, T, D)
    D = stacked.shape[2]
    t = np.arange(min_T)
    mean = stacked.mean(axis=0)   # (T, D)
    std  = stacked.std(axis=0)    # (T, D)

    ncols = min(5, D)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.5), squeeze=False)
    axes_flat = axes.flatten()

    cmap = plt.cm.tab10(np.linspace(0, 1, D))
    for i in range(D):
        ax = axes_flat[i]
        ax.plot(t, mean[:, i], color=cmap[i % 10], linewidth=1.2)
        ax.fill_between(t, mean[:, i] - std[:, i], mean[:, i] + std[:, i],
                        color=cmap[i % 10], alpha=0.25)
        ax.set_title(labels[i], fontsize=8)
        ax.set_xlabel("timestep", fontsize=7)

    for j in range(D, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{title}  (N={len(all_actions)} episodes, T=min {min_T})", color=TEXT_COLOR, fontsize=10)
    _apply_dark_style(fig, axes_flat[:D])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# per-episode processing
# ──────────────────────────────────────────────────────────────────────────────

def process_episode(hdf5_path: Path, out_dir: Path, make_plots: bool) -> np.ndarray | None:
    """Process one episode. Returns the action array (for aggregate plotting)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        # ── scan all numeric datasets ────────────────────────────────────────
        all_datasets = scan_hdf5(f)

        # ── main action array ────────────────────────────────────────────────
        if "action" not in f:
            print(f"  [SKIP] {hdf5_path.parent.name} — no 'action' dataset")
            return None
        action: np.ndarray = f["action"][:]             # (T, D)

        # qpos for reference
        qpos: np.ndarray | None = None
        if "observations/qpos" in f:
            qpos = f["observations/qpos"][:]

        # attrs
        action_format = f.attrs.get("action_format", None)
        action_layout = f.attrs.get("action_layout", None)
        if isinstance(action_layout, bytes):
            action_layout = action_layout.decode()
        if isinstance(action_format, bytes):
            action_format = action_format.decode()

    # ── dim labels & groups ──────────────────────────────────────────────────
    T, D = action.shape
    labels = _dim_labels(D, action_layout)
    groups = _dim_groups(D, action_layout)

    # ── CSV ──────────────────────────────────────────────────────────────────
    np.save(out_dir / "action.npy", action)
    col_names = labels
    csv_lines = [",".join(col_names)]
    for row in action:
        csv_lines.append(",".join(f"{v:.6f}" for v in row))
    (out_dir / "action.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # ── JSON summary ─────────────────────────────────────────────────────────
    summary: dict = {
        "episode": hdf5_path.stem,
        "hdf5_path": str(hdf5_path),
        "num_steps": T,
        "action_dim": D,
        "action_format": action_format,
        "action_layout": action_layout,
        "dim_labels": labels,
        # all numeric datasets in the file
        "all_numeric_datasets": all_datasets,
        # per-dim detailed stats
        "action_stats_per_dim": {
            labels[i]: {k: (v[i] if isinstance(v, list) else v)
                         for k, v in compute_stats(action).items()}
            for i in range(D)
        },
        # global stats (across all dims × all timesteps)
        "action_stats_global": compute_stats(action),
    }
    if qpos is not None:
        qpos_labels = [f"qpos_{i}" for i in range(qpos.shape[1])]
        summary["qpos_stats_per_dim"] = {
            qpos_labels[i]: {k: (v[i] if isinstance(v, list) else v)
                              for k, v in compute_stats(qpos).items()}
            for i in range(qpos.shape[1])
        }

    (out_dir / "action_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── plots ─────────────────────────────────────────────────────────────────
    if make_plots:
        ep_title = hdf5_path.parent.name
        plot_timeseries(action, labels, groups,
                        out_dir / "plot_timeseries.png",
                        f"{ep_title} — action time-series")
        plot_histograms(action, labels,
                        out_dir / "plot_histograms.png",
                        f"{ep_title} — per-dim distributions")
        plot_boxplots(action, labels,
                      out_dir / "plot_boxplots.png",
                      f"{ep_title} — boxplots")
        plot_correlation(action, labels,
                         out_dir / "plot_correlation.png",
                         f"{ep_title} — dim correlation")

    print(f"  [OK] {hdf5_path.parent.name}  T={T} D={D}  → {out_dir}")
    return action


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=str(DEFAULT_ROOT_DIR),
        help=f"Root directory (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "output_dir", nargs="?", default=None,
        help="Output base directory (default: <root_dir>/inspection_output)",
    )
    parser.add_argument(
        "--task", nargs="+", metavar="TASK", default=None,
        help="One or more task sub-directory names to inspect. "
             "If omitted, episode_* dirs are searched directly under root_dir.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib figure generation (JSON/CSV only).",
    )
    args = parser.parse_args()

    root    = Path(args.root_dir).resolve()
    out_base = Path(args.output_dir).resolve() if args.output_dir else root / "inspection_output"
    make_plots = not args.no_plots

    if args.task:
        episode_dirs = []
        for task_name in args.task:
            task_dir = root / task_name
            if not task_dir.is_dir():
                print(f"[WARN] Task directory not found: {task_dir}", file=sys.stderr)
                continue
            episode_dirs.extend(sorted(task_dir.glob("episode_*")))
        if not episode_dirs:
            print(f"[ERROR] No episode_* directories found for tasks: {args.task}", file=sys.stderr)
            sys.exit(1)
    else:
        episode_dirs = sorted(root.glob("episode_*"))
        if not episode_dirs:
            print(f"[ERROR] No episode_* directories found under {root}", file=sys.stderr)
            sys.exit(1)

    print(f"Root    : {root}")
    print(f"Tasks   : {args.task if args.task else '(flat episode_* search)'}")
    print(f"Output  : {out_base}")
    print(f"Episodes: {len(episode_dirs)}")
    print(f"Plots   : {'yes' if make_plots else 'no (--no-plots)'}")
    print()

    all_actions: list[np.ndarray] = []
    last_layout: str | None = None

    for ep_dir in episode_dirs:
        hdf5_files = list(ep_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"  [SKIP] {ep_dir.name} — no .hdf5 file found")
            continue
        if len(hdf5_files) > 1:
            print(f"  [WARN] {ep_dir.name} — multiple .hdf5 files, using first: {hdf5_files[0].name}")

        ep_out = (out_base / ep_dir.parent.name / ep_dir.name
                  if (args.task and ep_dir.parent != root)
                  else out_base / ep_dir.name)

        action = process_episode(hdf5_files[0], ep_out, make_plots)
        if action is not None:
            all_actions.append(action)
            # peek layout for aggregate
            if last_layout is None:
                with h5py.File(hdf5_files[0], "r") as f:
                    last_layout = f.attrs.get("action_layout", None)
                    if isinstance(last_layout, bytes):
                        last_layout = last_layout.decode()

    # ── aggregate cross-episode plot ─────────────────────────────────────────
    if make_plots and len(all_actions) > 1:
        D = all_actions[0].shape[1]
        agg_labels = _dim_labels(D, last_layout)
        print(f"\nGenerating aggregate plot for {len(all_actions)} episodes …")
        plot_aggregate(all_actions, agg_labels,
                       out_base / "aggregate_plot.png",
                       "Cross-episode mean ± 1σ")

        # also aggregate JSON stats
        min_T = min(a.shape[0] for a in all_actions)
        stacked = np.stack([a[:min_T] for a in all_actions], axis=0)
        agg_summary = {
            "num_episodes": len(all_actions),
            "action_dim": D,
            "dim_labels": agg_labels,
            "min_T": min_T,
            "per_dim_across_episodes": {
                agg_labels[i]: compute_stats(stacked[:, :, i].reshape(-1, 1))
                for i in range(D)
            },
        }
        (out_base / "aggregate_summary.json").write_text(
            json.dumps(agg_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
