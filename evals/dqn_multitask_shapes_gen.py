import os
import copy
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import DQNPolicy

from algos.sf_multitask import SFBase
from envs.shapes.multitask_shapes import MultitaskShapes, ShapesAttrCombFactory
from utils import setup_eval_paths, update_json_file
from yaml_utils import load_yaml
from algos.nets import ConcatActionValue
from algos.embedding_ops import (
    extract_bert_layer_embeddings,
    extract_elmo_layer_embeddings_tfhub,
    precompute_bert_embeddings,
    precompute_elmo_embeddings_tfhub,
)


def evaluate_single_seed(*, agent: SFBase, env_hparams: dict, store_path: str, seed: int, set_id: str) -> float:

    env_hparams_seeded = copy.deepcopy(env_hparams)
    env_hparams_seeded["seed"] = seed
    env_factory = ShapesAttrCombFactory(hparams=env_hparams_seeded, store_path=store_path)
    eval_env: MultitaskShapes = env_factory.get_env(set_id=set_id, purpose="EVAL")
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
            a_batch: Batch = agent(Batch(obs=[obs], info=[]))
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
                

def evaluation(store_path:str, model_path:str, config_path:str, precomp_path:str, set_id:str, n_seeds:int=10, n_eps:int=10) -> None:

    with open(config_path, "r") as f:
        hparams = load_yaml(f)

    exp_hparams = hparams["experiment"]
    env_hparams = hparams["environment"] if "environment" in hparams else hparams
    base_seed = hparams["general"]["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")
    
    precomp_model = exp_hparams["embedding_model"]
    layer_index = exp_hparams["layer_index"]

    if precomp_model in ("BERT", "bert"):
        embedding_path = os.path.join(precomp_path, 'bert_embeddings.pt')
    elif precomp_model in ("ELMO", "elmo"):
        embedding_path = os.path.join(precomp_path, 'elmo_embeddings.pt')
    else:
        raise ValueError(f"Model {precomp_model} not recognised. Must be either bert or elmo.") 
    
    precomp_embeddings = torch.load(embedding_path, map_location=device)
    if precomp_model in ("BERT", "bert"): 
        layer_embeddings = extract_bert_layer_embeddings(precomp_embeddings, layer_ind=layer_index)
    elif precomp_model in ("ELMO", "elmo"):
        layer_embeddings_np = extract_elmo_layer_embeddings_tfhub(embedding_dict=precomp_embeddings, layer_ind=layer_index)
        layer_embeddings = {
                instr: torch.from_numpy(arr).to(device).float() for instr, arr in layer_embeddings_np.items()
            }

    env_factory = ShapesAttrCombFactory(hparams=env_hparams, store_path=store_path)
    dummy_env = env_factory.get_env(set_id="TRAIN", purpose="EVAL")
    all_instructions = env_factory.get_all_instructions()
    
    emb_dim = layer_embeddings[next(iter(layer_embeddings))].shape[0]
    feat_dim = dummy_env.observation_space["features"].shape[0]
    in_dim = feat_dim + emb_dim

    nnet = ConcatActionValue(
        in_dim=in_dim,
        num_actions=int(dummy_env.action_space.n),
        h=exp_hparams["hidden_dim"],
        precom_embeddings=layer_embeddings,
        device=device
    )
    optim = torch.optim.Adam(nnet.parameters(), lr=exp_hparams["lr"])
    rb = ReplayBuffer(size=exp_hparams['buffer_size'])

    agent = DQNPolicy(
        model=nnet,
        optim=optim,
        is_double=False,
        action_space=dummy_env.action_space,
        discount_factor=env_hparams["disc_fact"],
        target_update_freq=exp_hparams["target_update_steps"]
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
            set_id=set_id
        )
        
        for skill_id, skill_stat in result_stat.items():
            if skill_id not in skill_stats:
                skill_stats[skill_id] = {}
            
            for stat_id, val in skill_stat.items():
                if stat_id not in skill_stats[skill_id]:
                    skill_stats[skill_id][stat_id] = []
                skill_stats[skill_id][stat_id].append(val)
    
    for skill_id, skill_stat in skill_stats.items():
        for stat_name, val in skill_stat.items():
            skill_stat[stat_name] = round(sum(val)/len(val), 2)
    

    return skill_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assess the performance of pre-trained SFBase.")
    parser.add_argument(
        "--run_id",
        type=str,
        # dqn_multitask_shapes_best_large_20250828_120149
        default="dqn_multitask_shapes_best_large_20250830_211616",
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
    parser.add_argument(
        "--set_id",
        type=str,
        default="TRAIN",
        help="Whether to evaluate on the TRAIN or HOLDOUT set.",
    )
    parser.add_argument(
        "--store_name",
        type=str,
        default="shapes_result_comp_train_debug",
        help="Name of the json file where results should be written. If file doesnt exist it will be created.",
    )
    
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    store_path, model_path, config_path, precomp_path, run_id = setup_eval_paths(
        script_path=script_path, run_id=args.run_id, run_path=args.run_path, last_model=args.last_model
    )

    result_dict = evaluation(
        store_path=store_path,
        model_path=model_path,
        config_path=config_path,
        precomp_path=precomp_path,
        n_seeds=args.n_seeds,
        set_id=args.set_id
    )

    results_pth = os.path.join(store_path, f"{args.store_name}.json")
    update_json_file(file_pth=results_pth, key=run_id, value=result_dict)
