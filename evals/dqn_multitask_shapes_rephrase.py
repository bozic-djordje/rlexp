import os
import copy
from typing import Dict, List
import numpy as np
import torch

from tianshou.data import Batch
from tianshou.policy import DQNPolicy

from algos.sf_multitask import SFBase
from envs.shapes.multitask_shapes import MultitaskShapes, ShapesAttrCombFactory, generate_instruction, create_synonyms, create_all_synonyms
from utils import setup_eval_paths, update_json_file
from yaml_utils import load_yaml
from algos.nets import (
    ConcatActionValue,
    extract_bert_layer_embeddings,
    extract_elmo_layer_embeddings_tfhub,
    precompute_bert_embeddings,
    precompute_elmo_embeddings_tfhub,
)


def evaluate_single_seed(*, agent: SFBase, env_hparams: Dict, synonyms: Dict, templates: List, store_path: str, seed: int) -> float:

    env_hparams_seeded = copy.deepcopy(env_hparams)
    env_hparams_seeded["seed"] = seed
    env_factory = ShapesAttrCombFactory(hparams=env_hparams_seeded, store_path=store_path)
    eval_env: MultitaskShapes = env_factory.get_env(set_id="TRAIN", purpose="EVAL")
    goal_list = eval_env.goal_list

    skill_stats = {}
    for goal in goal_list:
        instruction_synonyms = create_synonyms(goal=goal, templates=templates, synonyms=synonyms, use_features=env_hparams["use_features"])
        original_instruction = generate_instruction(instr=templates[0], goal=goal, all_feature_keys=env_hparams["use_features"])
        
        all_instructions = [original_instruction]
        all_instructions.extend(instruction_synonyms)

        for instruction in instruction_synonyms:
            obs, _ = eval_env.reset(options={"goal": goal})
            obs["instr"] = instruction
            ret = 0
            confounder_visited = 0
            steps = 0

            done = False
            while not done:
                a_batch: Batch = agent(Batch(obs=[obs], info=[]))
                action_id = a_batch.act.item()
                next_obs, reward, terminated, truncated, info = eval_env.step(action_id)
                obs = next_obs
                obs["instr"] = instruction
                done = terminated or truncated
                
                ret += reward
                steps += 1
                if "is_confounder" in info and info["is_confounder"]:
                    confounder_visited += 1
            
            goal_properly_reached = int(not confounder_visited)

            if original_instruction not in skill_stats:
                skill_stats[original_instruction] = {
                    "syn_return": [],
                    "syn_steps": [],
                    "syn_goal_first": [],
                    "original_return": [],
                    "original_steps": [],
                    "original_goal_first": []
                }
            
            if instruction == original_instruction:
                skill_stats[original_instruction]["original_return"].append(ret)
                skill_stats[original_instruction]["original_steps"].append(steps)
                skill_stats[original_instruction]["original_goal_first"].append(goal_properly_reached)
            else:
                skill_stats[original_instruction]["syn_return"].append(ret)
                skill_stats[original_instruction]["syn_steps"].append(steps)
                skill_stats[original_instruction]["syn_goal_first"].append(goal_properly_reached)
        
    for original_instruction, stat in skill_stats.items():
        skill_stats[original_instruction]["syn_return"] = np.round(np.mean(skill_stats[original_instruction]["syn_return"]), 2)
        skill_stats[original_instruction]["syn_steps"] = np.round(np.mean(skill_stats[original_instruction]["syn_steps"]), 2)
        skill_stats[original_instruction]["syn_goal_first"] = np.round(np.mean(skill_stats[original_instruction]["syn_goal_first"]), 2)

        skill_stats[original_instruction]["original_return"] = np.round(np.mean(skill_stats[original_instruction]["original_return"]), 2)
        skill_stats[original_instruction]["original_steps"] = np.round(np.mean(skill_stats[original_instruction]["original_steps"]), 2)
        skill_stats[original_instruction]["original_goal_first"] = np.round(np.mean(skill_stats[original_instruction]["original_goal_first"]), 2)
    
    return skill_stats
                

def evaluation(store_path:str, model_path:str, config_path:str, rephrase_path:str, precomp_path:str, n_seeds:int=10) -> None:

    with open(config_path, "r") as f:
        hparams = load_yaml(f)

    with open(rephrase_path, "r") as f:
        rephrase_hparams = load_yaml(f)
    
    exp_hparams = hparams["experiment"]
    env_hparams = hparams["environment"] if "environment" in hparams else hparams
    base_seed = hparams["general"]["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    precomp_model = exp_hparams["embedding_model"]
    layer_index = exp_hparams["layer_index"]

    env_factory = ShapesAttrCombFactory(hparams=env_hparams, store_path=store_path)
    dummy_env = env_factory.get_env(set_id="TRAIN", purpose="EVAL")

    _, synonyms_list = create_all_synonyms(
        goal_list=dummy_env.goal_list,
        templates=rephrase_hparams[env_hparams["task_id"]], 
        synonyms=rephrase_hparams["synonyms"],
        use_features=env_hparams["use_features"])
    
    if precomp_model in ("BERT", "bert"):
        embedding_path = os.path.join(precomp_path, 'bert_embeddings.pt')
    elif precomp_model in ("ELMO", "elmo"):
        embedding_path = os.path.join(precomp_path, 'elmo_embeddings.pt')
    else:
        raise ValueError(f"Model {precomp_model} not recognised. Must be either bert or elmo.") 
    
    loaded_embeddings = torch.load(embedding_path, map_location=device)

    if precomp_model in ("BERT", "bert"):
        precomp_embeddings = precompute_bert_embeddings(synonyms_list, device=device)
        
    elif precomp_model in ("ELMO", "elmo"):
        precomp_embeddings_np = precompute_elmo_embeddings_tfhub(synonyms_list)
        precomp_embeddings = {
            instr: torch.from_numpy(arr).to(device).float() for instr, arr in precomp_embeddings_np.items()
        }
    else:
        raise ValueError(f"Model {precomp_model} not recognised. Must be either bert or elmo.")
    
    if precomp_model in ("BERT", "bert"): 
        layer_embeddings = extract_bert_layer_embeddings(precomp_embeddings, layer_ind=layer_index)
        loaded_layer_embeddings = extract_bert_layer_embeddings(loaded_embeddings, layer_ind=layer_index)
    elif precomp_model in ("ELMO", "elmo"):
        layer_embeddings_np = extract_elmo_layer_embeddings_tfhub(embedding_dict=precomp_embeddings, layer_ind=layer_index)
        layer_embeddings = {
                instr: torch.from_numpy(arr).to(device).float() for instr, arr in layer_embeddings_np.items()
            }
        loaded_layer_embeddings_np = extract_elmo_layer_embeddings_tfhub(embedding_dict=loaded_embeddings, layer_ind=layer_index)
        loaded_layer_embeddings = {
                instr: torch.from_numpy(arr).to(device).float() for instr, arr in loaded_layer_embeddings_np.items()
            }
    
    for instr, _ in layer_embeddings.items():
        if instr in loaded_layer_embeddings:
            layer_embeddings[instr] = loaded_layer_embeddings[instr]

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

    agent.model.precomp_embed = layer_embeddings

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
            synonyms=rephrase_hparams["synonyms"],
            templates=rephrase_hparams[env_hparams["task_id"]]
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
        default="dqn_multitask_shapes_best_large_20250830_184952",
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
        default=2,
        help="Number of random seeds to evaluate (default: 10).",
    )
    parser.add_argument(
        "--last_model",
        type=bool,
        default=True,
        help="Whether to use the model from the last epoch or the best model.",
    )
    parser.add_argument(
        "--store_name",
        type=str,
        default="shapes_result_comp_rephrase_bert",
        help="Name of the json file where results should be written. If file doesnt exist it will be created.",
    )
    parser.add_argument(
        "--rephrase_config",
        type=str,
        default="shapes",
        help="Name of the config file defining rephrased instructions. Expected to be in configs directory.",
    )

    
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    store_path, model_path, config_path, precomp_path, run_id = setup_eval_paths(
        script_path=script_path, run_id=args.run_id, run_path=args.run_path, last_model=args.last_model
    )

    rephrase_fname = f"{args.rephrase_config}.yaml" if not args.rephrase_config.endswith(".yaml") else args.rephrase_config
    rephrase_config = os.path.join(os.path.dirname(script_path), "configs", rephrase_fname)

    result_dict = evaluation(
        store_path=store_path,
        model_path=model_path,
        config_path=config_path,
        n_seeds=args.n_seeds,
        rephrase_path=rephrase_config,
        precomp_path=precomp_path
    )

    results_pth = os.path.join(store_path, f"{args.store_name}.json")
    update_json_file(file_pth=results_pth, key=run_id, value=result_dict)
