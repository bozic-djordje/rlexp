#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

METRICS_ORDER = ["return", "steps", "goal_first"]
DPI = 150

def load_data(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_axes(json_data: Dict[str, Dict[str, Dict[str, float]]]):
    """
    Returns:
        models: List[str]
        instructions: List[str]
        metrics: List[str]
    """
    models = list(json_data.keys())

    # Union of instruction ids across models
    instr_set = set()
    for m in models:
        instr_set |= set(json_data[m].keys())
    instructions = sorted(instr_set)

    # Union of metrics across all model-instruction entries
    metric_set = set()
    for m in models:
        for instr in json_data[m]:
            metric_set |= set(json_data[m][instr].keys())
    # Keep preferred order where possible
    metrics = [m for m in METRICS_ORDER if m in metric_set] + [m for m in sorted(metric_set) if m not in METRICS_ORDER]
    return models, instructions, metrics

def values_for_metric(
    json_data: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    instructions: List[str],
    metric: str,
) -> np.ndarray:
    """
    Shape: (num_models, num_instructions)
    Missing values become np.nan.
    """
    mat = np.full((len(models), len(instructions)), np.nan, dtype=float)
    for mi, model in enumerate(models):
        for ii, instr in enumerate(instructions):
            if instr in json_data.get(model, {}) and metric in json_data[model][instr]:
                mat[mi, ii] = float(json_data[model][instr][metric])
    return mat

def plot_grouped_bars(
    data_matrix: np.ndarray,
    models: List[str],
    instructions: List[str],
    metric: str,
    out_png_path: str,
):
    """
    data_matrix: (num_models, num_instructions)
    Creates a single figure with grouped bars and saves it.
    """
    num_models, num_instr = data_matrix.shape
    x = np.arange(num_instr)

    # Bar width and offsets so groups are centered
    total_group_width = 0.8
    bar_width = total_group_width / max(1, num_models)
    offsets = (np.arange(num_models) - (num_models - 1) / 2) * bar_width

    plt.figure()
    for mi, model in enumerate(models):
        heights = data_matrix[mi]
        y = np.nan_to_num(heights, nan=0.0)
        plt.bar(x + offsets[mi], y, width=bar_width, label=model)

    plt.xticks(x, instructions, rotation=15, ha="right")
    plt.xlabel("Instruction")
    plt.ylabel(metric)
    plt.title(f"{metric} by instruction and model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=DPI)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Produce grouped bar plots (return, steps, goal_first) from a results JSON."
    )
    parser.add_argument(
        "--store_name",
        type=str,
        default="shapes_result_comp_eval",
        help="Name of the json file where results are stored.",
    )
    args = parser.parse_args()

    # Resolve paths
    base_name = args.store_name if args.store_name.endswith(".json") else f"{args.store_name}.json"
    artefacts_dir = os.path.join(os.path.dirname(__file__), "artefacts")
    input_path = os.path.join(artefacts_dir, base_name)

    # Output directory and filenames
    plots_dir = os.path.join(artefacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load and prep
    data = load_data(input_path)
    models, instructions, metrics = collect_axes(data)

    # Make one plot per metric, named after the json with metric suffix
    stem = os.path.splitext(base_name)[0]
    for metric in metrics:
        mat = values_for_metric(data, models, instructions, metric)
        out_png = os.path.join(plots_dir, f"{stem}_{metric}.png")
        plot_grouped_bars(mat, models, instructions, metric, out_png)

    print(f'Plots saved to: {plots_dir}')
    for metric in metrics:
        print(f" - {stem}_{metric}.png")

if __name__ == "__main__":
    main()
