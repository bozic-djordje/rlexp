#!/usr/bin/env python3
"""
Produce grouped bar plots (return, steps, goal_first) from a results JSON,
filtering and ordering models via a YAML config {model_folder: model_id}.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Preferred metric order for plotting (if present in data)
METRICS_ORDER = ["return", "steps", "goal_first"]
DPI = 150

# Predefined distinct base colors for up to ~10 models
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

# YAML loader (PyYAML)
try:
    import yaml
except Exception:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")


# ---------- Path helpers ----------

def _script_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _artefacts_dir() -> str:
    return os.path.join(_script_root(), "artefacts")


def _configs_dir() -> str:
    return os.path.join(_script_root(), "configs")


def _resolve_config_path(config_name: str) -> str:
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


# ---------- Data helpers ----------

def load_json_data(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_ordered_map(config_path: str) -> List[Tuple[str, str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping {model_folder: model_id}")
    return list(data.items())


def collect_axes(
    json_data: Dict[str, Dict[str, Dict[str, float]]],
    selected_models: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    models = [m for m in selected_models if m in json_data]
    instr_set = set()
    for m in models:
        instr_set |= set(json_data[m].keys())
    instructions = sorted(instr_set)
    metric_set = set()
    for m in models:
        for instr in json_data[m]:
            metric_set |= set(json_data[m][instr].keys())
    metrics = [m for m in METRICS_ORDER if m in metric_set] + \
              [m for m in sorted(metric_set) if m not in METRICS_ORDER]
    return models, instructions, metrics


def values_for_metric(
    json_data: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    instructions: List[str],
    metric: str,
) -> np.ndarray:
    mat = np.full((len(models), len(instructions)), np.nan, dtype=float)
    for mi, model in enumerate(models):
        for ii, instr in enumerate(instructions):
            if instr in json_data.get(model, {}) and metric in json_data[model][instr]:
                mat[mi, ii] = float(json_data[model][instr][metric])
    return mat


# ---------- Plotting ----------

def plot_grouped_bars(
    data_matrix: np.ndarray,
    legend_labels: List[str],
    instructions: List[str],
    metric: str,
    out_png_path: str,
):
    num_models, num_instr = data_matrix.shape
    x = np.arange(num_instr)
    total_group_width = 0.8
    bar_width = total_group_width / max(1, num_models)
    offsets = (np.arange(num_models) - (num_models - 1) / 2) * bar_width

    # Wider figure to give labels more space
    plt.figure(figsize=(max(10, num_instr * 0.6), 6))
    for mi, legend in enumerate(legend_labels):
        heights = data_matrix[mi]
        y = np.nan_to_num(heights, nan=0.0)
        color = MODEL_BASE_COLORS[mi % len(MODEL_BASE_COLORS)]
        plt.bar(x + offsets[mi], y, width=bar_width, label=legend, color=color)

    plt.xticks(x, instructions, rotation=30, ha="right")
    plt.xlabel("Instruction")
    plt.ylabel(metric)
    plt.title(f"{metric} by instruction and model")
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # add more space for x-labels
    plt.savefig(out_png_path, dpi=DPI)
    plt.close()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Produce grouped bar plots (return, steps, goal_first) from a results JSON, "
                    "filtering and ordering models via a YAML config."
    )
    parser.add_argument(
        "--store_name",
        type=str,
        default="sf_vs_dqn_best_model_holdout",
        help="Name of the JSON file under ./artefacts (with or without .json).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sf_vs_dqn_best_model_train_convergence",
        help="YAML config FILENAME (not a path) located in ./configs, mapping {model_folder: model_id}.",
    )
    args = parser.parse_args()

    base_name = args.store_name if args.store_name.endswith(".json") else f"{args.store_name}.json"
    artefacts_dir = _artefacts_dir()
    input_path = os.path.join(artefacts_dir, base_name)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Results JSON not found: {input_path}")

    config_path = _resolve_config_path(args.config)
    config_pairs = load_config_ordered_map(config_path)

    data = load_json_data(input_path)

    selected_models = [k for (k, _) in config_pairs]
    legend_labels = [v for (_, v) in config_pairs]

    missing = [m for m in selected_models if m not in data]
    if missing:
        print("[warn] These model_folders from config were not found in JSON and will be skipped:")
        for m in missing:
            print(f" - {m}")

    models, instructions, metrics = collect_axes(data, selected_models)
    label_map = dict(config_pairs)
    legend_labels_aligned = [label_map[m] for m in models]

    plots_dir = os.path.join(artefacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    stem = os.path.splitext(base_name)[0]
    if len(models) == 0:
        raise RuntimeError("No models to plot after filtering by config. Check your inputs.")

    for metric in metrics:
        mat = values_for_metric(data, models, instructions, metric)
        out_png = os.path.join(plots_dir, f"{stem}_{metric}.png")
        plot_grouped_bars(mat, legend_labels_aligned, instructions, metric, out_png)

    print(f"Plots saved to: {plots_dir}")
    for metric in metrics:
        print(f" - {stem}_{metric}.png")


if __name__ == "__main__":
    main()
