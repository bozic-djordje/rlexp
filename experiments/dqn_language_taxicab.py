import os
from typing import Dict
import torch
import yaml
import optuna, pickle

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.nets import ConcatActionValue, precompute_instruction_embeddings
from algos.common import EpsilonDecayHookFactory, SaveHookFactory
from envs.taxicab.language_taxicab import LanguageTaxicab, LanguageTaxicabFactory
from utils import setup_experiment


def experiment(trial: optuna.trial.Trial=None) -> float:
    script_path = os.path.abspath(__file__)
    experiment_name, store_path, yaml_path, precomp_path = setup_experiment(script_path=script_path)
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    exp_hparams = hparams["experiment"] if "experiment" in hparams else hparams
    env_hparams = hparams["environment"] if "environment" in hparams else hparams
    seed = hparams["general"]["seed"]

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

    # TODO: This will be a relatively common op, see if to extract it into the function.
    embedding_path = os.path.join(precomp_path, 'bert_embeddings.pt')
    if os.path.isfile(embedding_path):
        precomp_embeddings = torch.load(embedding_path, map_location=device)
    else:
        precomp_embeddings = precompute_instruction_embeddings(all_instructions, device=device)
        torch.save(precomp_embeddings, embedding_path)
    
    in_dim = train_env.observation_space["features"].shape[0] + precomp_embeddings[next(iter(precomp_embeddings))].shape[0]
    
    nnet = ConcatActionValue(
        in_dim=in_dim,
        num_actions=int(train_env.action_space.n),
        h=exp_hparams["hidden_dim"],
        precom_embeddings=precomp_embeddings,
        device=device
    )

    optim = torch.optim.Adam(nnet.parameters(), lr=exp_hparams["step_size"])
    rb = ReplayBuffer(size=exp_hparams['buffer_size'])

    agent = DQNPolicy(
        model=nnet, 
        optim=optim,
        action_space=train_env.action_space, 
        discount_factor=exp_hparams["disc_fact"], 
        target_update_freq=exp_hparams["target_update_steps"]
    )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
    test_collector = Collector(agent, test_env, exploration_noise=True)
    
    n_epochs = exp_hparams["n_epochs"]
    n_steps = exp_hparams["epoch_steps"]
    epoch_hook_factory = EpsilonDecayHookFactory(hparams=exp_hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)
    save_hook_factory = SaveHookFactory(save_path=f'{store_path}/best_model.pth')
    
    result = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=exp_hparams["step_per_collect"],
        update_per_step=exp_hparams["update_per_step"], episode_per_test=exp_hparams["episode_per_test"], batch_size=exp_hparams["batch_size"],
        train_fn=epoch_hook_factory.hook,
        test_fn=lambda epoch, global_step: agent.set_eps(exp_hparams["test_epsilon"]),
        save_best_fn=save_hook_factory.hook,
        logger=logger
    ).run()
    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')


if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    _, _, yaml_path, _ = setup_experiment(script_path=script_path)
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
    exp_hparams = hparams["experiment"]
    single_experiment = True
    for key, val in exp_hparams.items():
        if isinstance(val, list) and key != "hidden_dim":
            single_experiment = False
        
    if not single_experiment:
        study = optuna.create_study(direction="maximize",
                                    pruner=optuna.pruners.MedianPruner(
                                            n_startup_trials=5, n_warmup_steps=5))
        study.optimize(experiment, n_trials=50, timeout=4*3600)
        print("Best trial:", study.best_trial.number, study.best_params)
        with open("optuna_study.pkl", "wb") as f: pickle.dump(study, f)
    else:
        experiment()
    
    