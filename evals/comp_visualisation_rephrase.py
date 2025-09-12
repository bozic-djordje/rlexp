#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------- Constants -----------

METRICS_ORDER = ["return", "steps", "goal_first"]
VARIANTS = ["original", "syn"]
DPI = 150

# Predefined distinct base colors for up to ~10 models
MODEL_BASE_COLORS = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# YAML loader (PyYAML)
try:
    import yaml
except Exception:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")


# ----------- Path helpers -----------

def _script_root() -> str:
    """Directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def _artefacts_dir() -> str:
    """./artefacts next to this script."""
    return os.path.join(_script_root(), "artefacts")


def _configs_dir() -> str:
    """./configs next to this script."""
    return os.path.join(_script_root(), "configs")


def _resolve_config_path(config_name: str) -> str:
    """
    Resolve YAML config filename (not path) inside ./configs.
    Accepts 'name', 'name.yaml', or 'name.yml'.
    """
    cfg_dir = _configs_dir()
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
        f"Config '{config_name}' not found in '{cfg_dir}'. Tried: {', '.join(candidates)}"
    )


# ----------- Data helpers -----------

def load_data(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_ordered_map(config_path: str) -> List[Tuple[str, str]]:
    """
    Load YAML mapping {model_folder: model_id} preserving order.
    Returns list of (model_folder, model_id) in YAML order.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping {model_folder: model_id}")
    return list(data.items())  # insertion order preserved (Py3.7+)


def collect_axes(
    json_data: Dict[str, Dict[str, Dict[str, float]]],
    selected_models: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
        models: filtered & ordered list (per config) present in json_data
        instructions: sorted list of all instructions across selected models
        metrics: ordered list of base metrics present (METRICS_ORDER first)
    """
    models = [m for m in selected_models if m in json_data]

    instr_set = set()
    for m in models:
        instr_set |= set(json_data[m].keys())
    instructions = sorted(instr_set)

    present = set()
    for m in models:
        for instr in json_data[m]:
            keys = json_data[m][instr].keys()
            for base in METRICS_ORDER:
                if (
                    f"original_{base}" in keys or
                    f"syn_{base}" in keys or
                    (base == "goal_first" and (
                        "original_goals_first" in keys or "syn_goals_first" in keys))
                ):
                    present.add(base)

    metrics = [m for m in METRICS_ORDER if m in present]
    return models, instructions, metrics


def _resolve_key(d: Dict[str, float], prefix: str, metric: str) -> float:
    """
    Robustly fetch <prefix>_<metric> with a fallback for 'goals_first' typo.
    Returns np.nan if missing.
    """
    key = f"{prefix}_{metric}"
    if key in d:
        return float(d[key])
    if metric == "goal_first":
        alt = f"{prefix}_goals_first"  # tolerate plural 'goals'
        if alt in d:
            return float(d[alt])
    return np.nan


def values_for_metric_with_variants(
    json_data: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    instructions: List[str],
    metric: str,
) -> np.ndarray:
    """
    Shape: (num_variants=2, num_models, num_instructions)
    Order of variants: ['original', 'syn'].
    Missing values become np.nan.
    """
    num_variants = len(VARIANTS)
    mat = np.full((num_variants, len(models), len(instructions)), np.nan, dtype=float)
    for vi, variant in enumerate(VARIANTS):
        for mi, model in enumerate(models):
            for ii, instr in enumerate(instructions):
                entry = json_data.get(model, {}).get(instr, {})
                mat[vi, mi, ii] = _resolve_key(entry, variant, metric)
    return mat


# ----------- Plotting -----------

def plot_grouped_bars_with_variants(
    data_tensor: np.ndarray,
    legend_labels: List[str],
    instructions: List[str],
    metric: str,
    out_png_path: str,
):
    """
    data_tensor: (num_variants, num_models, num_instructions)
    legend_labels: labels for legend (model_ids from YAML), length == num_models
    """
    num_variants, num_models, num_instr = data_tensor.shape
    x = np.arange(num_instr)

    total_group_width = 0.9
    slot_width = total_group_width / max(1, num_models)
    bar_width = slot_width / max(1, num_variants)

    model_offsets = (np.arange(num_models) - (num_models - 1) / 2) * slot_width
    variant_offsets = (np.arange(num_variants) - (num_variants - 1) / 2) * bar_width

    plt.figure(figsize=(max(10, num_instr * 0.6), 6))
    legend_handles = []

    for mi in range(num_models):
        base_color = MODEL_BASE_COLORS[mi % len(MODEL_BASE_COLORS)]
        # lighter for original, darker for syn
        original_color = mcolors.to_rgba(base_color, alpha=0.6)
        syn_color = mcolors.to_rgba(base_color, alpha=1.0)

        for vi, variant in enumerate(VARIANTS):
            heights = data_tensor[vi, mi]
            y = np.nan_to_num(heights, nan=0.0)
            positions = x + model_offsets[mi] + variant_offsets[vi]
            color = original_color if variant == "original" else syn_color
            plt.bar(positions, y, width=bar_width, color=color)

        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker="s", linestyle="",
                markersize=10,
                markerfacecolor=syn_color,  # show the darker (syn) color in legend
                markeredgecolor="none",
                label=legend_labels[mi],
            )
        )

    plt.xticks(x, instructions, rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by instruction and model (original=lighter, syn=darker)")

    plt.legend(
        handles=legend_handles,
        ncol=4, fontsize="small", loc="upper center",
        bbox_to_anchor=(0.5, -0.15), frameon=False
    )

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------- Main -----------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Produce grouped bar plots (return, steps, goal_first) with original/syn bars per model, "
            "filtering and ordering models via a YAML config {model_folder: model_id}."
        )
    )
    parser.add_argument(
        "--store_name",
        type=str,
        default="sf_vs_dqn_best_rephrase",
        help="Name of the JSON file under ./artefacts (with or without .json).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bert_vs_elmo_vs_dqn_rephrase",
        help="YAML config FILENAME (not a path) in ./configs, mapping {model_folder: model_id}.",
    )
    args = parser.parse_args()

    # Resolve input JSON path under ./artefacts
    base_name = args.store_name if args.store_name.endswith(".json") else f"{args.store_name}.json"
    artefacts_dir = _artefacts_dir()
    input_path = os.path.join(artefacts_dir, base_name)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Results JSON not found: {input_path}")

    # Resolve config path and load ordered mapping
    config_path = _resolve_config_path(args.config)
    config_pairs = load_config_ordered_map(config_path)  # List[(model_folder, model_id)]

    # Load JSON results
    data = load_data(input_path)

    # Prepare selected models (order preserved) and legend labels (model_ids)
    selected_models = [k for (k, _) in config_pairs]
    legend_labels = [v for (_, v) in config_pairs]

    # Warn about any models in config missing from JSON
    missing = [m for m in selected_models if m not in data]
    if missing:
        print("[warn] These model_folders from config were not found in JSON and will be skipped:")
        for m in missing:
            print(f"  - {m}")

    # Collect axes using only models that exist in the JSON
    models, instructions, metrics = collect_axes(data, selected_models)

    if len(models) == 0:
        raise RuntimeError("No models to plot after filtering by config. Check your inputs.")

    # Align legend_labels to the filtered 'models'
    label_map = dict(config_pairs)  # model_folder -> model_id
    legend_labels_aligned = [label_map[m] for m in models]

    # Output directory
    plots_dir = os.path.join(artefacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # One plot per metric, named after the json with metric suffix
    stem = os.path.splitext(base_name)[0]
    for metric in metrics:
        tensor = values_for_metric_with_variants(data, models, instructions, metric)
        out_png = os.path.join(plots_dir, f"{stem}_{metric}.png")
        plot_grouped_bars_with_variants(tensor, legend_labels_aligned, instructions, metric, out_png)

    print(f"Plots saved to: {plots_dir}")
    for metric in metrics:
        print(f" - {stem}_{metric}.png")


if __name__ == "__main__":
    main()
