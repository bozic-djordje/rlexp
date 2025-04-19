import os
from typing import Dict
import torch
import math

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, Collector, ReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tqdm import tqdm
from algos.nets import FCActionValue
from algos.common import TrainHookFactory
from envs.taxicab.single_taxicab import FeatureTaxicab
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
        },
        {
            "colour": "green",
            "building": "office",
            "size": "small",
            "fill": "outlined"
        }
    ]

    train_env = FeatureTaxicab(
        hparams=hparams,
        location_features=location_features,
        origin_ind=1,
        dest_ind=2,
        store_path=store_path
    )

    test_env = FeatureTaxicab(
        hparams=hparams,
        location_features=location_features,
        origin_ind=1,
        dest_ind=2,
        store_path=store_path
    )
    
    nnet = FCActionValue(
        in_dim=train_env.observation_space.shape[0],
        num_actions=int(train_env.action_space.n),
        h=[768]
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
    TrainHookFactory.initialize(hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)

    result = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=10,
        update_per_step=1, episode_per_test=100, batch_size=64,
        train_fn=TrainHookFactory.train_hook,
        test_fn=lambda epoch, global_step: agent.set_eps(0.05),
        logger=logger
    ).run()