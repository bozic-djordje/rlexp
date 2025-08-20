import os
import copy
import numpy as np
import torch

from tianshou.data import Batch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.common import GroupedReplayBuffer
from algos.sf_multitask import SFBase
from envs.shapes.multitask_shapes import MultitaskShapes, ShapesAttrCombFactory, ShapesPositionFactory
from utils import setup_eval_paths, setup_experiment
from yaml_utils import load_yaml
from algos.nets import (
    precompute_bert_embeddings,
    extract_bert_layer_embeddings,
    FCTrunk,
    FCTree,
)


def evaluate_single_seed(*, agent: SFBase, env_hparams: dict, store_path: str, seed: int, n_episode:int=10) -> float:

    env_hparams_seeded = copy.deepcopy(env_hparams)
    env_hparams_seeded["seed"] = seed
    env_factory = ShapesPositionFactory(hparams=env_hparams_seeded, store_path=store_path)
    eval_env: MultitaskShapes = env_factory.get_env(set_id="TRAIN", purpose="EVAL")
    goal_list = eval_env.goal_list

    skill_stats = {}
    for goal in goal_list:

        obs, _ = eval_env.reset(options={"goal": goal})
        ret = 0
        confounder_visited = 0
        steps = 0

        instr = obs["instr"]

        done = False
        while not done:
            a_batch: Batch = agent(Batch(obs=[obs]))
            action_id = a_batch.act.item()
            next_obs, reward, terminated, truncated, info = eval_env.step(action_id)
            obs = next_obs
            done = terminated or truncated
            
            ret += reward
            steps += 1
            if "is_confounder" in info and info["is_confounder"]:
                confounder_visited += 1
        
        goal_properly_reached = int(not confounder_visited) and steps < 15

        if instr not in skill_stats:
            skill_stats[instr] = {
                "return": [],
                "steps": [],
                "goal_first": []
            }
        
        skill_stats[instr]["return"].append(ret)
        skill_stats[instr]["steps"].append(steps)
        skill_stats[instr]["goal_first"].append(goal_properly_reached)
    
    for instr, stat in skill_stats.items():
        skill_stats[instr]["return"] = np.round(np.mean(skill_stats[instr]["return"]), 2)
        skill_stats[instr]["steps"] = np.round(np.mean(skill_stats[instr]["steps"]), 2)
        skill_stats[instr]["goal_first"] = np.round(np.mean(skill_stats[instr]["goal_first"]), 2)
    
    return skill_stats
                

def evaluation(store_path:str, model_path:str, config_path:str, precomp_path:str, n_seeds:int=10, n_eps:int=10) -> None:

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

    env_factory = ShapesAttrCombFactory(hparams=env_hparams, store_path=store_path)
    dummy_env = env_factory.get_env(set_id="TRAIN", purpose="EVAL")
    all_instructions = env_factory.get_all_instructions()

    phi_nn = FCTree(
        in_dim=dummy_env.observation_space["features"].shape,
        num_heads=dummy_env.action_space.n,
        h_trunk=exp_hparams["phi_trunk_dim"],
        h_head=exp_hparams["phi_head_dim"],
        device=device
    )

    psi_nn = FCTrunk(
        in_dim=exp_hparams["phi_head_dim"][-1] if isinstance(exp_hparams["phi_head_dim"], list) else exp_hparams["phi_head_dim"],
        h=exp_hparams["psi_nn_dim"] if isinstance(exp_hparams["psi_nn_dim"], list) else [exp_hparams["psi_nn_dim"]],
        device=device
    )
    
    # TODO: Implement PrioritisedGroupedReplayBuffer to access prioritised memory replay
    rb = GroupedReplayBuffer(size=exp_hparams['buffer_size'])
    
    agent = SFBase(
        phi_nn=phi_nn, 
        psi_nn=psi_nn,
        dec_nn=None,
        rb=rb,
        num_skills=len(all_instructions),
        action_space=dummy_env.action_space,
        precomp_embeddings=layer_embeddings,
        l2_freq_scaling=exp_hparams["l2_freq_scaling"],
        phi_lr=exp_hparams["phi_lr"],
        psi_lr=exp_hparams["psi_lr"],
        psi_lambda=exp_hparams["psi_lambda"],
        phi_lambda=exp_hparams["phi_lambda"],
        psi_update_tau=exp_hparams["psi_update_tau"],
        phi_update_tau=exp_hparams["phi_update_tau"],
        phi_update_ratio=exp_hparams["phi_update_ratio"],
        gamma=env_hparams["disc_fact"],
        seed=base_seed,
        device=device
    )
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    agent.set_eps(exp_hparams["test_epsilon"])

    # free dummy env now that specs are captured
    dummy_env.close()

    skill_stats = {}

    for i in range(n_seeds):
        seed = base_seed + i
        np.random.seed(seed)
        torch.manual_seed(seed)

        result_stat = evaluate_single_seed(
            agent=agent,
            env_hparams=env_hparams,
            store_path=store_path,
            seed=seed,
            n_episode=n_eps
        )
        
        for skill_id, skill_stat in result_stat.items():
            if skill_id not in skill_stats:
                skill_stats[skill_id] = {}
            
            for stat_id, val in skill_stat.items():
                if stat_id not in skill_stats[skill_id]:
                    skill_stats[skill_id][stat_id] = []
                skill_stats[skill_id][stat_id].append(val)

        
        # TODO: See which stat to print
        # print(f"[EVAL] Seed {seed:<3d} — mean return: {result_stat:.3f}")

    # TODO: See how to create a bar plot here
    # mean_of_means = float(np.mean(results))
    # std_of_means = float(np.std(results))

    # writer.add_scalar("eval/mean_return_avg", mean_of_means, n_seeds)
    # writer.add_scalar("eval/mean_return_std", std_of_means, n_seeds)

    # print("--------------------------------------------------------------")
    # print(f"[EVAL] Across {n_seeds} seeds — mean: {mean_of_means:.3f} | std: {std_of_means:.3f}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assess the performance of pre-trained SFBase.")
    parser.add_argument(
        "--run_id",
        type=str,
        default="sf_multitask_shapes_best_20250819_223238",
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
        "--last_model",
        type=bool,
        default=True,
        help="Whether to use the model from the last epoch or the best model.",
    )
    
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    store_path, model_path, config_path, precomp_path = setup_eval_paths(
        script_path=script_path, run_id=args.run_id, run_path=args.run_path, last_model=args.last_model
    )

    evaluation(
        store_path=store_path,
        model_path=model_path,
        config_path=config_path,
        precomp_path=precomp_path,
        n_seeds=args.n_seeds
    )
