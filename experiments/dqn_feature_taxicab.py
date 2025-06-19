import os
from typing import Dict
import torch
import math

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.nets import FCActionValue
from algos.common import EpsilonDecayHook, SaveHook
from envs.taxicab.feature_taxicab import FeatureTaxicab
from utils import setup_experiment, setup_artefact_paths
from yaml_utils import load_yaml, save_yaml


if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    store_path, config_path = setup_artefact_paths(script_path=script_path)
    experiment_name, store_path, _ = setup_experiment(store_path=store_path, config_path=config_path)
    
    import yaml
    with open(config_path, 'r') as file:
        hparams = load_yaml(file)
    exp_hparams = hparams["experiment"]
    env_hparams = hparams["environment"]
    
    writer = SummaryWriter(store_path)
    logger = TensorboardLogger(writer)

    location_features = [
        {
            "colour": "red",
            "building": "hospital",
            "size": "big",
            "fill": "filled"
        },
        {
            "colour": "green",
            "building": "office",
            "size": "small",
            "fill": "outlined"
        },
        {
            "colour": "blue",
            "building": "school",
            "size": "big",
            "fill": "filled"
        },
        {
            "colour": "yellow",
            "building": "library",
            "size": "big",
            "fill": "filled"
        }
    ]

    train_env = FeatureTaxicab(
        hparams=env_hparams,
        location_features=location_features,
        store_path=store_path,
        easy_mode=env_hparams["easy_mode"] if "easy_mode" in env_hparams else False
    )

    test_env = FeatureTaxicab(
        hparams=env_hparams,
        location_features=location_features,
        store_path=store_path,
        easy_mode=env_hparams["easy_mode"] if "easy_mode" in env_hparams else False
    )
    
    nnet = FCActionValue(
        in_dim=train_env.observation_space.shape[0],
        num_actions=int(train_env.action_space.n),
        h=exp_hparams["hidden_dim"]
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
    epoch_hook_factory = EpsilonDecayHook(hparams=exp_hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)
    save_hook_factory = SaveHook(save_path=f'{store_path}/best_model.pth')
    
    result = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=200,
        update_per_step=0.25, episode_per_test=100, batch_size=exp_hparams["batch_size"],
        train_fn=epoch_hook_factory.hook,
        test_fn=lambda epoch, global_step: agent.set_eps(0.05),
        save_best_fn=save_hook_factory.hook,
        logger=logger
    ).run()
    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')