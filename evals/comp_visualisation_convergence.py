#!/usr/bin/env python3
"""
Plot convergence curves from TensorBoard event files for multiple models.

Key behavior changes:
- Output filename matches YAML config filename (e.g., compare_models.yaml -> compare_models.png).
- --config is a FILENAME ONLY from sibling 'configs/'.
- Default output dir is sibling 'artefacts/plots/' if --output_dir not provided.
- Default --root is '<script_dir>/../experiments/paper'.
- NEW: --alpha controls line transparency (default 0.7).
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as ea

# YAML reader (PyYAML required)
try:
    import yaml
except Exception:
    print("[error] PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise

MODEL_BASE_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def _script_root() -> str:
    """Directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def _default_root() -> str:
    """Default root: '<script_dir>/../experiments/paper'."""
    return os.path.normpath(os.path.join(_script_root(), "..", "experiments", "paper"))


def _resolve_config_path(config_name: str) -> str:
    """
    Resolve the YAML config file path from '<script_dir>/configs/'.
    Accept 'name', 'name.yaml', or 'name.yml'.
    """
    base_dir = _script_root()
    cfg_dir = os.path.join(base_dir, "configs")

    candidates = []
    if config_name.lower().endswith((".yaml", ".yml")):
        candidates.append(os.path.join(cfg_dir, config_name))
    else:
        candidates.append(os.path.join(cfg_dir, config_name + ".yaml"))
        candidates.append(os.path.join(cfg_dir, config_name + ".yml"))

    for p in candidates:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"Could not find config '{config_name}' in '{cfg_dir}'. "
        f"Tried: {', '.join(candidates)}"
    )


def _default_output_dir() -> str:
    """Default output directory: '<script_dir>/artefacts/plots'."""
    return os.path.join(_script_root(), "artefacts", "plots")


def find_event_dirs(root: str) -> Dict[str, List[str]]:
    """Recursively find directories under `root` that contain TensorBoard event files.
    Returns: basename(dir) -> list of full dir paths with event files.
    """
    result: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root):
        if any(fn.startswith("events.out.tfevents") or fn.startswith("events.tfevents")
               for fn in filenames):
            base = os.path.basename(dirpath.rstrip(os.sep))
            result.setdefault(base, []).append(dirpath)
    return result


def pick_latest_dir(paths: List[str]) -> str:
    """Choose the directory with the most recent event file mtime."""
    best_path = None
    best_mtime = -1.0
    for p in paths:
        mt = _dir_latest_event_mtime(p)
        if mt > best_mtime:
            best_mtime = mt
            best_path = p
    return best_path


def _dir_latest_event_mtime(path: str) -> float:
    latest = -1.0
    try:
        for fn in os.listdir(path):
            if fn.startswith("events.out.tfevents") or fn.startswith("events.tfevents"):
                mtime = os.path.getmtime(os.path.join(path, fn))
                if mtime > latest:
                    latest = mtime
    except Exception:
        pass
    return latest


def load_scalar_series(run_dir: str, scalar_tag: str) -> Tuple[List[int], List[float]]:
    """Load (steps, values) for `scalar_tag` from TensorBoard event files in `run_dir`."""
    acc = ea.EventAccumulator(
        run_dir,
        size_guidance={
            ea.SCALARS: 0,
            ea.HISTOGRAMS: 0,
            ea.IMAGES: 0,
            ea.AUDIO: 0,
            ea.COMPRESSED_HISTOGRAMS: 0,
        },
    )
    acc.Reload()

    if scalar_tag not in acc.Tags().get("scalars", []):
        raise KeyError(
            f"Metric '{scalar_tag}' not found in {run_dir}. "
            f"Available: {sorted(acc.Tags().get('scalars', []))}"
        )

    scalar_list = acc.Scalars(scalar_tag)
    steps = [s.step for s in scalar_list]
    values = [float(s.value) for s in scalar_list]
    return steps, values


def smooth_ema(values: List[float], smoothing: float) -> List[float]:
    """TensorBoard-style exponential smoothing."""
    if not values:
        return values
    if smoothing <= 0.0:
        return values[:]
    y = [values[0]]
    a = smoothing
    for x in values[1:]:
        y.append(a * y[-1] + (1.0 - a) * x)
    return y


def main():
    parser = argparse.ArgumentParser(description="Plot convergence curves from TensorBoard runs.")
    parser.add_argument(
        "--root",
        type=str,
        default=_default_root(),
        help="Root directory containing (nested) TensorBoard event files. "
             "Defaults to '<script_dir>/../experiments/paper'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sf_vs_dqn_best_model_train_convergence",
        help="YAML config FILENAME (not a path). File must be in the sibling 'configs/' directory.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="train/returns_stat/mean",
        help="Scalar tag to plot (default: 'train/returns_stat/mean').",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.75,
        help="Exponential smoothing factor in [0,1). 0 disables smoothing. Default: 0.75.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Line transparency in [0,1]. Lower is more transparent. Default: 0.7.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the plot. Defaults to sibling 'artefacts/plots/'.",
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["step", "index"],
        default="step",
        help="X-axis: 'step' (default, TB step) or 'index' (sample index).",
    )
    args = parser.parse_args()

    # Resolve config path from sibling 'configs/' dir
    try:
        config_path = _resolve_config_path(args.config)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory (default to sibling 'artefacts/plots/')
    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Derive output filename from the config filename (replace extension with .png)
    cfg_base = os.path.basename(config_path)
    base_no_ext = os.path.splitext(cfg_base)[0]
    output_path = os.path.join(output_dir, f"{base_no_ext}.png")

    # Discover all event directories under root
    print(f"[info] Scanning for event files under: {args.root}")
    name_to_dirs = find_event_dirs(args.root)
    if not name_to_dirs:
        print("[error] No TensorBoard event files found.", file=sys.stderr)
        sys.exit(2)

    # Load model selections from YAML
    with open(config_path, "r", encoding="utf-8") as f:
        selection = yaml.safe_load(f)
    if not isinstance(selection, dict):
        print("[error] --config file must contain a dict: {model_folder: legend_tag}", file=sys.stderr)
        sys.exit(3)

    # Prepare plot
    plt.figure(figsize=(9, 5))
    color_iter = iter(MODEL_BASE_COLORS)

    missing_models = []
    plotted = 0

    for model_folder, legend_tag in selection.items():
        if model_folder not in name_to_dirs:
            missing_models.append(model_folder)
            continue

        # If multiple matches for the same basename, pick the most recent
        candidates = name_to_dirs[model_folder]
        run_dir = candidates[0] if len(candidates) == 1 else pick_latest_dir(candidates)

        try:
            steps, values = load_scalar_series(run_dir, args.metric)
        except KeyError as e:
            print(f"[warn] {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[warn] Failed to load {model_folder} from {run_dir}: {e}", file=sys.stderr)
            continue

        if not steps or not values:
            print(f"[warn] Empty series for {model_folder} ({run_dir}).", file=sys.stderr)
            continue

        y = smooth_ema(values, args.smoothing)
        x = steps if args.x_axis == "step" else list(range(len(y)))

        try:
            color = next(color_iter)
        except StopIteration:
            print("[error] Ran out of MODEL_BASE_COLORS; increase palette size.", file=sys.stderr)
            sys.exit(4)

        # Use configurable transparency
        plt.plot(x, y, label=legend_tag, linewidth=2.0, alpha=args.alpha, color=color)
        plotted += 1

    if missing_models:
        print(
            "[warn] Could not find these model folders under --root (by basename): "
            + ", ".join(missing_models),
            file=sys.stderr,
        )

    if plotted == 0:
        print("[error] Nothing was plotted (no valid metrics found).", file=sys.stderr)
        sys.exit(5)

    plt.xlabel("Step" if args.x_axis == "step" else "Index")
    plt.ylabel(args.metric)
    plt.title("Convergence")
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    print(f"[info] Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
