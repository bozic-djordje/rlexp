import re
import os
from copy import deepcopy
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from algos.nets import extract_bert_layer_embeddings, precompute_bert_embeddings, precompute_elmo_embeddings_tfhub, extract_elmo_layer_embeddings_tfhub
from envs.shapes.multitask_shapes import MultitaskShapes, generate_instruction, ShapesAttrCombFactory
from utils import setup_artefact_paths
from tqdm import tqdm


def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt((x * x).sum(axis=axis, keepdims=True)).clip(min=eps)
    return x / norm


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,D), b: (M,D), both L2-normalized → returns (N,M) cosine sims."""
    return a @ b.T


def plot_layer_panels(
    layers: List[int],
    per_layer_within: Dict[int, np.ndarray],
    per_layer_non: Dict[int, np.ndarray],
    suptitle: str,
    out_path: str,
    bins: int = 40
) -> None:
    """Side-by-side subplots (one per layer) showing within-synonym vs others."""
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4.5), dpi=150, squeeze=False)
    axes = axes[0]

    for i, layer in enumerate(layers):
        ax = axes[i]
        syn_sims = per_layer_within.get(layer, np.array([]))
        non_sims = per_layer_non.get(layer, np.array([]))

        # histograms
        if syn_sims.size > 0:
            ax.hist(syn_sims, bins=bins, density=True, alpha=0.55,
                    label=f"Within-synonym (n={len(syn_sims)})")
        if non_sims.size > 0:
            ax.hist(non_sims, bins=bins, density=True, alpha=0.55,
                    label=f"Synonym vs others (n={len(non_sims)})")

        # KDE if scipy present
        try:
            from scipy.stats import gaussian_kde
            vals = []
            if syn_sims.size > 1: vals.append(syn_sims)
            if non_sims.size > 1: vals.append(non_sims)
            if len(vals) > 0:
                x_min = float(min(v.min() for v in vals))
                x_max = float(max(v.max() for v in vals))
                xs = np.linspace(x_min, x_max, 250)
                if syn_sims.size > 1:
                    kde_syn = gaussian_kde(syn_sims)
                    ax.plot(xs, kde_syn(xs), linewidth=2, label="KDE (within)")
                if non_sims.size > 1:
                    kde_non = gaussian_kde(non_sims)
                    ax.plot(xs, kde_non(xs), linewidth=2, label="KDE (others)")
        except Exception:
            pass

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Cosine similarity")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(loc="best")

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def create_synonyms(env, goals: List, synonyms: Dict) -> Dict: 
    synonyms_dict = defaultdict(set) 
    synonyms_list = set() 
    for goal in goals: 
        goal_tuple = (goal['colour'], goal['shape'])
        for template in env._instr_templates:
            instr = generate_instruction(instr=deepcopy(template), goal=deepcopy(goal), all_feature_keys=env._features.keys())
            for keyword, synonyms in hparams["synonyms"].items():
                if keyword in instr: 
                    instr_2 = deepcopy(instr)
                    for synonym in synonyms: 
                        instr_3 = deepcopy(instr_2) 
                        final_instr = instr_3.replace(keyword, synonym) 
                        synonyms_dict[goal_tuple].add(final_instr) 
                        synonyms_list.add(final_instr) 
    return synonyms_dict, list(synonyms_list)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Plot cosine similarity hist/KDE per goal across multiple layers.")
    parser.add_argument("--model_name", type=str, default="elmo",
                        help="Which model to use to extract embeddings.")
    parser.add_argument("--config_name", type=str, default="shapes",
                        help="Experiment name which needs to match config file name.")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1],
                        help="List of BERT layers to plot (default: [0, 7, 11]).")
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, config_name=args.config_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    env_factory = ShapesAttrCombFactory(hparams=hparams, store_path=store_path)
    env: MultitaskShapes = env_factory.get_env(set_id='TRAIN')

    # Build synonyms per goal
    synonyms_dict, synonyms_list = create_synonyms(env=env, goals=env.goal_list, synonyms=hparams["synonyms"])

    # Paths
    precomp_path = os.path.join(store_path, 'precomputed')
    os.makedirs(precomp_path, exist_ok=True)

    plots_dir = os.path.join(store_path, "plots", "embeddings")
    os.makedirs(plots_dir, exist_ok=True)

    if args.model_name == "bert":
        # Precompute embeddings for the full set we need
        print("Precomputing BERT embeddings for all instructions (all layers retained)...")
        embedding_path = os.path.join(precomp_path, 'bert_embeddings.pt')
        # Always recompute or load if you prefer caching:
        precomp_embeddings = precompute_bert_embeddings(synonyms_list, device=device)
        torch.save(precomp_embeddings, embedding_path)
    else:
        print("Precomputing ELMO embeddings for all instructions (all layers retained)...")
        embedding_path = os.path.join(precomp_path, 'elmo_embeddings.pt')
        # Always recompute or load if you prefer caching:
        precomp_embeddings = precompute_elmo_embeddings_tfhub(synonyms_list)
    torch.save(precomp_embeddings, embedding_path)

    # Corpus order (keep only items that have embeddings)
    # Extract per-layer embeddings into matrices aligned with 'corpus'
    # Build one common corpus covering all layers
    sample_layer = args.layers[0]
    if args.model_name == "bert":
        sample_layer_dict: Dict[str, torch.Tensor] = extract_bert_layer_embeddings(
            embedding_dict=precomp_embeddings, layer_ind=sample_layer
        )
    else:
        sample_layer_dict: Dict[str, torch.Tensor] = extract_elmo_layer_embeddings_tfhub(
            embedding_dict=precomp_embeddings, layer_ind=sample_layer
        )
    
    corpus = [s for s in synonyms_list if s in sample_layer_dict]
    if not corpus:
        raise RuntimeError("No instructions found in extracted layer embeddings. Check your keys.")

    instr_to_idx = {txt: i for i, txt in enumerate(corpus)}

    # Build matrices for each requested layer (N, D), L2-normalized
    layer_to_matrix: Dict[int, np.ndarray] = {}
    for layer in args.layers:
        if args.model_name == "bert":
            layer_dict: Dict[str, torch.Tensor] = extract_bert_layer_embeddings(
                embedding_dict=precomp_embeddings, layer_ind=layer
            )
        else:
            layer_dict: Dict[str, torch.Tensor] = extract_elmo_layer_embeddings_tfhub(
                embedding_dict=precomp_embeddings, layer_ind=layer
            )
        emb_list = []
        missing = 0
        for instr in corpus:
            if instr not in layer_dict:
                missing += 1
                continue
            emb = layer_dict[instr]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().float().cpu().numpy()
            else:
                emb = np.asarray(emb, dtype=np.float32)
            emb_list.append(emb)
        if missing > 0:
            print(f"[WARN] Layer {layer}: {missing} corpus items missing embeddings; they were skipped.")

        all_embs = np.vstack(emb_list) if len(emb_list) > 0 else np.zeros((0, 1), dtype=np.float32)
        all_embs = l2_normalize_np(all_embs, axis=1)
        layer_to_matrix[layer] = all_embs

    # Rebuild instr_to_idx and corpus in case any item was missing for some layer:
    # We keep the original 'corpus' for indexing, but when computing similarities we will mask
    # out indices beyond the available matrix length for each layer (to be safe).
    N = len(corpus)

    # ----------------------
    # Per-goal multi-layer plots
    # ----------------------
    print("Computing similarities and plotting per-goal panels...")
    for (colour, shape), group_instrs in tqdm(synonyms_dict.items()):
        # Deduplicate & keep only those present in corpus (embedded)
        seen = set()
        group_instrs = [s for s in group_instrs if s in instr_to_idx and not (s in seen or seen.add(s))]
        if len(group_instrs) < 2:
            continue

        group_indices = np.array([instr_to_idx[s] for s in group_instrs], dtype=np.int64)

        per_layer_within: Dict[int, np.ndarray] = {}
        per_layer_non: Dict[int, np.ndarray] = {}

        for layer in args.layers:
            E = layer_to_matrix[layer]
            if E.shape[0] == 0:
                per_layer_within[layer] = np.array([], dtype=np.float32)
                per_layer_non[layer] = np.array([], dtype=np.float32)
                continue

            # Filter indices to those < E.shape[0] (robustness if some items were missing)
            gi = group_indices[group_indices < E.shape[0]]
            if gi.size < 2:
                per_layer_within[layer] = np.array([], dtype=np.float32)
                per_layer_non[layer] = np.array([], dtype=np.float32)
                continue

            group_embs = E[gi]  # (G, D)
            sim_mat = cosine_similarity_matrix(group_embs, group_embs)
            iu = np.triu_indices(len(gi), k=1)
            within_syn = sim_mat[iu]

            all_indices = np.arange(E.shape[0])
            mask = np.ones(E.shape[0], dtype=bool)
            mask[gi] = False
            non_indices = all_indices[mask]
            syn_vs_non = cosine_similarity_matrix(group_embs, E[non_indices]).ravel() if non_indices.size > 0 else np.array([], dtype=np.float32)

            per_layer_within[layer] = within_syn
            per_layer_non[layer] = syn_vs_non

        goal_id = f"{colour}_{shape}"
        suptitle = f"Cosine similarity — {colour} {shape}"
        out_file = os.path.join(plots_dir, safe_filename(f"{goal_id}_layers_{'_'.join(map(str, args.layers))}.png"))
        plot_layer_panels(args.layers, per_layer_within, per_layer_non, suptitle, out_file, bins=40)

    # ----------------------
    # Final “All goals (averaged)” multi-layer plot
    # ----------------------
    print("Aggregating across all goals and plotting the final averaged figure...")
    agg_per_layer_within: Dict[int, List[np.ndarray]] = {l: [] for l in args.layers}
    agg_per_layer_non: Dict[int, List[np.ndarray]] = {l: [] for l in args.layers}

    for (colour, shape), group_instrs in synonyms_dict.items():
        seen = set()
        group_instrs = [s for s in group_instrs if s in instr_to_idx and not (s in seen or seen.add(s))]
        if len(group_instrs) < 2:
            continue
        group_indices = np.array([instr_to_idx[s] for s in group_instrs], dtype=np.int64)

        for layer in args.layers:
            E = layer_to_matrix[layer]
            if E.shape[0] == 0:
                continue
            gi = group_indices[group_indices < E.shape[0]]
            if gi.size < 2:
                continue

            group_embs = E[gi]
            sim_mat = cosine_similarity_matrix(group_embs, group_embs)
            iu = np.triu_indices(len(gi), k=1)
            within_syn = sim_mat[iu]

            all_indices = np.arange(E.shape[0])
            mask = np.ones(E.shape[0], dtype=bool)
            mask[gi] = False
            non_indices = all_indices[mask]
            syn_vs_non = cosine_similarity_matrix(group_embs, E[non_indices]).ravel() if non_indices.size > 0 else np.array([], dtype=np.float32)

            if within_syn.size > 0:
                agg_per_layer_within[layer].append(within_syn)
            if syn_vs_non.size > 0:
                agg_per_layer_non[layer].append(syn_vs_non)

    # Concatenate per layer
    final_within = {l: (np.concatenate(agg_per_layer_within[l]) if len(agg_per_layer_within[l]) > 0 else np.array([], dtype=np.float32))
                    for l in args.layers}
    final_non = {l: (np.concatenate(agg_per_layer_non[l]) if len(agg_per_layer_non[l]) > 0 else np.array([], dtype=np.float32))
                 for l in args.layers}

    final_title = f"Cosine similarity — ALL GOALS (layers: {', '.join(map(str, args.layers))})"
    final_path = os.path.join(plots_dir, safe_filename(f"ALL_GOALS_layers_{'_'.join(map(str, args.layers))}.png"))
    plot_layer_panels(args.layers, final_within, final_non, final_title, final_path, bins=40)

    print(f"Done. Plots saved to: {plots_dir}")
