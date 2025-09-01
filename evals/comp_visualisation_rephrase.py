#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

METRICS_ORDER = ["return", "steps", "goal_first"]
VARIANTS = ["original", "syn"]
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

def load_data(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_axes(json_data: Dict[str, Dict[str, Dict[str, float]]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
        models: List[str]
        instructions: List[str]
        metrics: List[str] (subset of METRICS_ORDER, if present)
    """
    models = list(json_data.keys())

    # Union of instruction ids across models
    instr_set = set()
    for m in models:
        instr_set |= set(json_data[m].keys())
    instructions = sorted(instr_set)

    # Decide which metrics are present (look for either original_ or syn_ prefixes)
    present = set()
    for m in models:
        for instr in json_data[m]:
            keys = json_data[m][instr].keys()
            for base in METRICS_ORDER:
                if (f"original_{base}" in keys or f"syn_{base}" in keys or
                    (base == "goal_first" and (
                        "original_goals_first" in keys or "syn_goals_first" in keys))):
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


def plot_grouped_bars_with_variants(
    data_tensor: np.ndarray,
    models: List[str],
    instructions: List[str],
    metric: str,
    out_png_path: str,
):
    num_variants, num_models, num_instr = data_tensor.shape
    x = np.arange(num_instr)

    total_group_width = 0.9
    slot_width = total_group_width / max(1, num_models)
    bar_width = slot_width / max(1, num_variants)

    model_offsets = (np.arange(num_models) - (num_models - 1) / 2) * slot_width
    variant_offsets = (np.arange(num_variants) - (num_variants - 1) / 2) * bar_width

    plt.figure(figsize=(max(10, num_instr * 0.6), 6))
    legend_handles = []

    for mi, model in enumerate(models):
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

        model_id = model.split("_")[0]
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker="s", linestyle="",
                markersize=10,
                markerfacecolor=syn_color,
                markeredgecolor="none",
                label=model_id
            )
        )

    plt.xticks(x, instructions, rotation=30, ha="right")
    # plt.xlabel("Instruction")
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



def main():
    parser = argparse.ArgumentParser(
        description="Produce grouped bar plots (return, steps, goal_first) with original/syn bars per model."
    )
    parser.add_argument(
        "--store_name",
        type=str,
        default="shapes_result_comp_rephrase_bert",
        help="Name of the json file inside artefacts (with or without .json).",
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
        tensor = values_for_metric_with_variants(data, models, instructions, metric)
        out_png = os.path.join(plots_dir, f"{stem}_{metric}.png")
        plot_grouped_bars_with_variants(tensor, models, instructions, metric, out_png)

    print(f'Plots saved to: {plots_dir}')
    for metric in metrics:
        print(f" - {stem}_{metric}.png")

if __name__ == "__main__":
    main()
