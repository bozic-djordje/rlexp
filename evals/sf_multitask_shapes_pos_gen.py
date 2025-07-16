import os
import copy
import numpy as np
import torch

from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.sf_bert import SFBert
from envs.shapes.multitask_shapes import MultitaskShapes, ShapesPositionFactory
from utils import setup_eval_paths, setup_experiment
from yaml_utils import load_yaml
from algos.nets import (
    precompute_bert_embeddings,
    extract_bert_layer_embeddings,
    FCTrunk,
    FCTree,
)


def evaluate_single_seed(*, agent: SFBert, env_hparams: dict, store_path: str, seed: int, n_episode:int=100) -> float:

    env_hparams_seeded = copy.deepcopy(env_hparams)
    env_hparams_seeded["seed"] = seed
    env_factory = ShapesPositionFactory(hparams=env_hparams_seeded, store_path=store_path)
    eval_env: MultitaskShapes = env_factory.get_env(set_id="HOLDOUT")
    eval_env._resample_interval = 1  # always new tasks

    collector = Collector(agent, eval_env, exploration_noise=False)
    collector.reset()
    result = collector.collect(n_episode=n_episode)

    mean_return = float(np.mean(result.returns))

    eval_env.close()
    del collector, eval_env

    return mean_return


def evaluation(store_path:str, model_path:str, config_path:str, precomp_path:str, n_seeds:int=10, n_eps:int=100) -> None:

    with open(config_path, "r") as f:
        hparams = load_yaml(f)

    exp_hparams = hparams["experiment"]
    env_hparams = hparams["environment"] if "environment" in hparams else hparams
    base_seed = hparams["general"]["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    writer = SummaryWriter(store_path)
    logger = TensorboardLogger(writer)

    
    embedding_path = os.path.join(precomp_path, "bert_embeddings.pt")
    precomp_embeddings = torch.load(embedding_path, map_location=device)
    
    bert_layer_ind = exp_hparams.get("bert_layer_index", -1)
    layer_embeddings = extract_bert_layer_embeddings(precomp_embeddings, layer_ind=bert_layer_ind)

    dummy_env = ShapesPositionFactory(hparams=env_hparams, store_path=store_path).get_env(set_id="TRAIN")

    phi_nn = FCTree(
        in_dim=dummy_env.observation_space["features"].shape,
        num_heads=dummy_env.action_space.n,
        h_trunk=exp_hparams["phi_trunk_dim"],
        h_head=exp_hparams["phi_head_dim"],
        device=device,
    )

    psi_nn = FCTrunk(
        in_dim=(
            exp_hparams["phi_head_dim"][-1]
            if isinstance(exp_hparams["phi_head_dim"], list)
            else exp_hparams["phi_head_dim"]
        ),
        h=(
            exp_hparams["psi_nn_dim"]
            if isinstance(exp_hparams["psi_nn_dim"], list)
            else [exp_hparams["psi_nn_dim"]]
        ),
        device=device,
    )

    if exp_hparams["prioritised_replay"]:
        rb = PrioritizedReplayBuffer(
            size=exp_hparams["buffer_size"],
            alpha=exp_hparams["priority_alpha"],
            beta=exp_hparams["priority_beta_start"],
        )
    else:
        rb = ReplayBuffer(size=exp_hparams["buffer_size"])

    agent = SFBert(
        phi_nn=phi_nn,
        psi_nn=psi_nn,
        dec_nn=None,
        rb=rb,
        action_space=dummy_env.action_space,
        precomp_embeddings=layer_embeddings,
        l2_freq_scaling=exp_hparams["l2_freq_scaling"],
        lr=exp_hparams["step_size"],
        target_update_freq=exp_hparams["target_update_steps"],
        cycle_update_freq=exp_hparams["cycle_update_steps"],
        gamma=env_hparams["disc_fact"],
        seed=base_seed,  # initial seed; overwritten per‑loop for torch/np rngs
        device=device,
    )
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    agent.set_eps(exp_hparams["test_epsilon"])

    # free dummy env now that specs are captured
    dummy_env.close()

    mean_returns = []

    for i in range(n_seeds):
        seed = base_seed + i
        np.random.seed(seed)
        torch.manual_seed(seed)

        mean_r = evaluate_single_seed(
            agent=agent,
            env_hparams=env_hparams,
            store_path=store_path,
            seed=seed,
            n_episode=n_eps
        )
        mean_returns.append(mean_r)
        print(f"[EVAL] Seed {seed:<3d} — mean return: {mean_r:.3f}")

    mean_of_means = float(np.mean(mean_returns))
    std_of_means = float(np.std(mean_returns))

    writer.add_scalar("eval/mean_return_avg", mean_of_means, n_seeds)
    writer.add_scalar("eval/mean_return_std", std_of_means, n_seeds)

    print("--------------------------------------------------------------")
    print(f"[EVAL] Across {n_seeds} seeds — mean: {mean_of_means:.3f} | std: {std_of_means:.3f}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained SF‑BERT agent over multiple seeds.")
    parser.add_argument(
        "--run_id",
        type=str,
        default="sf_multitask_shapes_mdp_slow_update_hard_20250715_163200",
        help="Run name of the model to be evaluated.",
    )
    parser.add_argument(
        "--run_path",
        type=str,
        default=None,
        help="Path to the directory containing the model to be evaluated.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of random seeds to evaluate (default: 10).",
    )
    parser.add_argument(
        "--n_eps",
        type=int,
        default=100,
        help="Number of episodes to evaluate for for each seed.",
    )
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    store_path, model_path, config_path, precomp_path = setup_eval_paths(
        script_path=script_path, run_id=args.run_id, run_path=args.run_path
    )

    evaluation(
        store_path=store_path,
        model_path=model_path,
        config_path=config_path,
        precomp_path=precomp_path,
        n_seeds=args.n_seeds,
        n_eps=args.n_eps
    )
