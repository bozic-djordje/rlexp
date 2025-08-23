import os
import numpy as np
import torch
import optuna, pickle

from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.sf_multitask import SFBase, SFMix
from algos.common import BetaAnnealHook, CompositeHook, EpsilonDecayHook, GroupedReplayBuffer, SFTrainer, SaveHook, TestFnHook
from envs.shapes.multitask_shapes import MultitaskShapes, ShapesAttrCombFactory
from utils import iterate_hyperparams, setup_artefact_paths, setup_experiment, setup_study, sample_hyperparams
from yaml_utils import load_yaml, save_yaml
from algos.nets import ScalarMix, precompute_bert_embeddings, extract_bert_layer_embeddings, FCTrunk, FCTree


def experiment(store_path:str, config_path:str, trial:optuna.trial.Trial=None, exact_hparams=None) -> float:
    _, store_path, precomp_path = setup_experiment(store_path=store_path, config_path=config_path)
    with open(config_path, 'r') as file:
        hparams = load_yaml(file)

    # We may perform a search on experiment hyper-parameters using Optuna
    exp_hparams = hparams["experiment"]
    # TODO: Make this less hacky! 1/2
    exp_hparams["resample_episodes"] = hparams["environment"]["resample_episodes"]
    
    if trial is not None:
        exp_hparams, only_sampled_hparams = sample_hyperparams(trial=trial, hparams=exp_hparams)
    elif exact_hparams is not None:
        exp_hparams = exact_hparams
    
    # Environment hyper-parameters are fixed
    env_hparams = hparams["environment"] if "environment" in hparams else hparams
    # TODO: Make this less hacky! 2/2
    env_hparams["resample_episodes"] = exp_hparams["resample_episodes"]
    seed = hparams["general"]["seed"]
    
    sampled_config = {
        "general": hparams["general"],
        "experiment": exp_hparams, 
        "environment": env_hparams
    }
    sampled_config_pth = os.path.join(store_path, os.path.basename(config_path))
    save_yaml(data=sampled_config, path=sampled_config_pth)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    writer = SummaryWriter(store_path)
    logger = TensorboardLogger(writer)

    env_factory = ShapesAttrCombFactory(
        hparams=env_hparams, 
        store_path=store_path
    )
    train_env: MultitaskShapes = env_factory.get_env(set_id='TRAIN')
    # TODO: Measure success on holdout later, for now success is measured on TRAIN (we are not measuring generalisation)
    test_env: MultitaskShapes = env_factory.get_env(set_id='TRAIN', purpose='EVAL')
    
    all_instructions = env_factory.get_all_instructions()

    embedding_path = os.path.join(precomp_path, 'bert_embeddings.pt')
    if os.path.isfile(embedding_path):
        precomp_embeddings = torch.load(embedding_path, map_location=device)
    else:
        precomp_embeddings = precompute_bert_embeddings(all_instructions, device=device)
        torch.save(precomp_embeddings, embedding_path)
    
    if "bert_layer_index" in exp_hparams:
        bert_layer_ind = exp_hparams["bert_layer_index"]
    else:
        bert_layer_ind = -1
    if not isinstance(bert_layer_ind, list):
        layer_embeddings = extract_bert_layer_embeddings(embedding_dict=precomp_embeddings, layer_ind=bert_layer_ind)
    else:
        layer_embeddings = precomp_embeddings
    
    # TODO: Implement PrioritisedGroupedReplayBuffer to access prioritised memory replay
    rb = GroupedReplayBuffer(size=exp_hparams['buffer_size'])
    
    phi_nn = FCTree(
        in_dim=train_env.observation_space["features"].shape,
        num_heads=train_env.action_space.n,
        h_trunk=exp_hparams["phi_trunk_dim"],
        h_head=exp_hparams["phi_head_dim"],
        device=device
    )

    psi_nn = FCTrunk(
        in_dim=exp_hparams["phi_head_dim"][-1] if isinstance(exp_hparams["phi_head_dim"], list) else exp_hparams["phi_head_dim"],
        h=exp_hparams["psi_nn_dim"] if isinstance(exp_hparams["psi_nn_dim"], list) else [exp_hparams["psi_nn_dim"]],
        device=device
    )
    
    if not isinstance(bert_layer_ind, list):
        agent = SFBase(
            phi_nn=phi_nn, 
            psi_nn=psi_nn,
            dec_nn=None,
            rb=rb,
            num_skills=len(all_instructions),
            action_space=train_env.action_space,
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
            seed=seed,
            device=device
        )
    else:
        mix_nn = ScalarMix(len(bert_layer_ind), trainable=True).to(device)
        agent = SFMix(
            phi_nn=phi_nn, 
            psi_nn=psi_nn,
            mix_nn=mix_nn,
            layers_to_mix=bert_layer_ind,
            dec_nn=None,
            rb=rb,
            num_skills=len(all_instructions),
            action_space=train_env.action_space,
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
            seed=seed,
            device=device
        )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
    train_collector.reset()
    train_collector.collect(n_step=exp_hparams["warmup_steps"], random=True)

    test_collector = Collector(agent, test_env, exploration_noise=True)
    test_rewards_history = []
    
    n_epochs = exp_hparams["n_epochs"]
    n_steps = exp_hparams["epoch_steps"]
    
    hooks = CompositeHook(agent=agent, logger=logger, hooks=[])
    epoch_hook = EpsilonDecayHook(hparams=exp_hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger, is_linear=exp_hparams["linear_schedule"])
    hooks.add_hook(epoch_hook)

    save_hook = SaveHook(save_path=f'{store_path}/best_model.pth')
    test_fn = TestFnHook(
        agent=agent, 
        logger=logger, 
        epsilon=exp_hparams["test_epsilon"],
        episodes_per_test=exp_hparams["episode_per_test"], 
        collector=test_collector, 
        history=test_rewards_history
    )
    
    _ = SFTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=exp_hparams["step_per_collect"],
        update_per_step=exp_hparams["update_per_step"], episode_per_test=exp_hparams["episode_per_test"], batch_size=exp_hparams["batch_size"],
        train_fn=hooks.hook,
        test_fn=test_fn.hook,
        save_best_fn=save_hook.hook,
        logger=logger
    ).run()
    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')

    if len(test_rewards_history) < 10:
        best_10_consec_avg = np.mean(test_rewards_history)
    else:
        cumsum = np.cumsum([0] + test_rewards_history)
        best_10_consec_avg = max((cumsum[i + 10] - cumsum[i]) / 10 for i in range(len(test_rewards_history) - 9))
    
    if trial is not None:
        writer.add_hparams(
            hparam_dict=only_sampled_hparams,
            metric_dict={
                "train/best_result": best_10_consec_avg
            }
        )
    elif exact_hparams is not None:
        writer.add_hparams(
            hparam_dict=exact_hparams,
            metric_dict={
                "train/best_result": best_10_consec_avg
            }
        )
    return best_10_consec_avg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run SF Multitask Shapes experiment.")
    parser.add_argument("--config_name", type=str, default=None, help="Experiment name which needs to match config file name.")
    parser.add_argument("--iterate_key", type=str, default=None, help="Parameter name through which to iterate. Must be in config and must be a list.")
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    store_path, config_path = setup_artefact_paths(script_path=script_path, config_name=args.config_name)
    
    with open(config_path, 'r') as file:
        hparams = load_yaml(file)
    
    exp_hparams = hparams["experiment"]
    n_trials = exp_hparams["n_trials"]

    # Automatically determine if we are running a single experiment or multiple
    # If multiple, Optuna will be used.
    single_experiment = True
    special_keys = {"phi_nn_dim", "phi_head_dim", "phi_trunk_dim", "psi_nn_dim", "float_keys", "log_domain_keys"}
    
    if args.iterate_key is not None:
        special_keys.add(args.iterate_key)

    # Multiple experiments warranted if there are hyper-parameters with multiple values
    for key, val in exp_hparams.items():
        if isinstance(val, list) and key not in special_keys:
            single_experiment = False   
        
    if not single_experiment:
        import functools
        
        if args.iterate_key is not None:
            raise ValueError(f"Cannot both iterate key {args.iterate_key} and run Optuna trial. Either remove --iterate_key argument, or have a single set of hyper-parameters!") 
        
        print(f'Optuna: Running {exp_hparams["n_trials"]} experiments to determine the best set of hyper-parameters.')
        _, store_path = setup_study(store_path=store_path, config_path=config_path)
        objective = functools.partial(experiment, store_path=store_path, config_path=config_path)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print("Best trial:", study.best_trial.number, study.best_params)
        
        if args.config_name is None:
            config_name = os.path.basename(config_path).split(".")[0]
        else:
            config_name = args.config_name
        optuna_path = os.path.join(store_path, f"{config_name}.pkl")
        with open(optuna_path, "wb") as f: pickle.dump(study, f)
    else:
        if args.iterate_key is None:
            print(f'Running a single experiment.')
            experiment(trial=None, store_path=store_path, config_path=config_path, exact_hparams=None)
        else:
            with open(config_path, 'r') as file:
                hparams = load_yaml(file)
            _, store_path = setup_study(store_path=store_path, config_path=config_path)

            exp_hparams = hparams["experiment"]
            exp_hparams["resample_episodes"] = hparams["environment"]["resample_episodes"]

            if not isinstance(exp_hparams[args.iterate_key], list):
                raise ValueError(f"Cannot iterate key {args.iterate_key}, key value not a list!") 
            
            hparams_list = iterate_hyperparams(hparams=exp_hparams, key=args.iterate_key)
            print(f'Iterating over key={args.iterate_key}. Running {len(hparams_list)} experiments.')
            for hp in hparams_list:
                experiment(trial=None, store_path=store_path, config_path=config_path, exact_hparams=hp)