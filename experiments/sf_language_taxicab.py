from copy import deepcopy
import os
import torch
import optuna, pickle

from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.sf_multitask import SFBase
from algos.common import BetaAnnealHook, CompositeHook, EpsilonDecayHook, SaveHook
from envs.taxicab.language_taxicab import LanguageTaxicab, LanguageTaxicabFactory
from utils import setup_artefact_paths, setup_experiment, setup_study, sample_hyperparams
from yaml_utils import load_yaml, save_yaml
from algos.nets import precompute_bert_embeddings, extract_bert_layer_embeddings, FCMultiHead, FCTrunk


def experiment(trial: optuna.trial.Trial, store_path:str, config_path:str) -> float:
    _, store_path, precomp_path = setup_experiment(store_path=store_path, config_path=config_path)
    with open(config_path, 'r') as file:
        hparams = load_yaml(file)

    # We may perform a search on experiment hyper-parameters using Optuna
    exp_hparams = hparams["experiment"]
    if trial is not None:
        exp_hparams = sample_hyperparams(trial=trial, hparams=exp_hparams)
    # Environment hyper-parameters are fixed
    env_hparams = hparams["environment"] if "environment" in hparams else hparams
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

    env_factory = LanguageTaxicabFactory(
        hparams=env_hparams, 
        store_path=store_path
    )
    train_env: LanguageTaxicab = env_factory.get_env(set_id='TRAIN')
    test_env: LanguageTaxicab = env_factory.get_env(set_id='HOLDOUT')
    
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
    layer_embeddings = extract_bert_layer_embeddings(embedding_dict=precomp_embeddings, layer_ind=bert_layer_ind)
    
    if exp_hparams['prioritised_replay'] is False:
        rb = ReplayBuffer(size=exp_hparams['buffer_size'])
    else:
        rb = PrioritizedReplayBuffer(
            size=exp_hparams['buffer_size'], 
            alpha=exp_hparams["priority_alpha"], 
            beta=exp_hparams["priority_beta_start"]
        )
    
    phi_nn = FCMultiHead(
        in_dim=train_env.observation_space["features"].shape[0],
        num_heads=train_env.action_space.n,
        h=exp_hparams["phi_nn_dim"],
        device=device
    )

    psi_nn = FCTrunk(
        in_dim=exp_hparams["phi_nn_dim"][-1],
        h=exp_hparams["psi_nn_dim"],
        device=device
    )

    if exp_hparams["use_reconstruction_loss"]:
        # Reverse layers of the phi_nn (which acts as an encoder)
        hidden_dim = deepcopy(exp_hparams["phi_nn_dim"])
        hidden_dim.reverse()
        in_dim = hidden_dim.pop(0)
        hidden_dim.append(train_env.observation_space["features"].shape[0])
        
        dec_nn = FCTrunk(
            in_dim=in_dim,
            h=hidden_dim,
            device=device
        )
    else:
        dec_nn = None
    
    agent = SFBase(
        phi_nn=phi_nn, 
        psi_nn=psi_nn,
        dec_nn=dec_nn,
        rb=rb,
        action_space=train_env.action_space,
        precomp_embeddings=layer_embeddings,
        l2_freq_scaling=exp_hparams["l2_freq_scaling"],
        lr=exp_hparams["step_size"],
        psi_update_tau=exp_hparams["target_update_steps"],
        phi_update_ratio=exp_hparams["cycle_update_steps"],
        gamma=env_hparams["disc_fact"],
        seed=seed,
        device=device
    )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
    test_collector = Collector(agent, test_env, exploration_noise=True)
    
    n_epochs = exp_hparams["n_epochs"]
    n_steps = exp_hparams["epoch_steps"]
    
    hooks = CompositeHook(agent=agent, logger=logger, hooks=[])
    epoch_hook = EpsilonDecayHook(hparams=exp_hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)
    hooks.add_hook(epoch_hook)

    if exp_hparams["prioritised_replay"]:
        beta_anneal_hook = BetaAnnealHook(
            agent=agent, 
            buffer=rb, 
            beta_start=exp_hparams["priority_beta_start"],
            beta_end=exp_hparams["priority_beta_end"], 
            frac=exp_hparams["priority_beta_frac"],
            max_steps=n_epochs*n_steps,
            logger=logger
        )
        hooks.add_hook(beta_anneal_hook)

    save_hook = SaveHook(save_path=f'{store_path}/best_model.pth')
    
    result = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=exp_hparams["step_per_collect"],
        update_per_step=exp_hparams["update_per_step"], episode_per_test=exp_hparams["episode_per_test"], batch_size=exp_hparams["batch_size"],
        train_fn=hooks.hook,
        test_fn=lambda epoch, global_step: agent.set_eps(exp_hparams["test_epsilon"]),
        save_best_fn=save_hook.hook,
        logger=logger
    ).run()
    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')
    return result.best_reward


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run SF Language Taxicab experiment.")
    parser.add_argument("--config_name", type=str, default=None, help="Experiment name which needs to match config file name.")
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
    special_keys = {"phi_nn_dim", "psi_nn_dim", "float_keys", "log_domain_keys"}
    # Multiple experiments warranted if there are hyper-parameters with multiple values
    for key, val in exp_hparams.items():
        if isinstance(val, list) and key not in special_keys:
            single_experiment = False   
        
    if not single_experiment:
        import functools
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
        print(f'Running a single experiment.')
        experiment(trial=None, store_path=store_path, config_path=config_path)