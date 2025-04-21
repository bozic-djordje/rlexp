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
from algos.common import EpsilonDecayHookFactory, SaveHookFactory
from envs.taxicab.feature_taxicab import FeatureTaxicab
from utils import setup_experiment, setup_artefact_paths


if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    experiment_name, store_path, yaml_path = setup_experiment(script_path=script_path)
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
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
        hparams=hparams,
        location_features=location_features,
        store_path=store_path
    )

    test_env = FeatureTaxicab(
        hparams=hparams,
        location_features=location_features,
        store_path=store_path
    )
    
    nnet = FCActionValue(
        in_dim=train_env.observation_space.shape[0],
        num_actions=int(train_env.action_space.n),
        h=hparams["hidden_dim"]
    )

    optim = torch.optim.Adam(nnet.parameters(), lr=hparams["step_size"])
    rb = ReplayBuffer(size=hparams['buffer_size'])
    
    agent = DQNPolicy(
        model=nnet, 
        optim=optim,
        action_space=train_env.action_space, 
        discount_factor=hparams["disc_fact"], 
        target_update_freq=hparams["target_update_steps"]
    )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
    test_collector = Collector(agent, test_env, exploration_noise=True)
    
    n_epochs = hparams["n_epochs"]
    n_steps = hparams["epoch_steps"]
    epoch_hook_factory = EpsilonDecayHookFactory(hparams=hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)
    save_hook_factory = SaveHookFactory(save_path=f'{store_path}/best_model.pth')
    
    result = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=200,
        update_per_step=0.25, episode_per_test=100, batch_size=hparams["batch_size"],
        train_fn=epoch_hook_factory.hook,
        test_fn=lambda epoch, global_step: agent.set_eps(0.05),
        save_best_fn=save_hook_factory.hook,
        logger=logger
    ).run()
    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')