import os
import torch
import gymnasium as gym

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.nets import FCActionValue
from algos.common import EpsilonDecayHookFactory
from utils import setup_experiment


if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    experiment_name, store_path, yaml_path, _ = setup_experiment(script_path=script_path)
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
    writer = SummaryWriter(store_path)
    logger = TensorboardLogger(writer)

    train_env = gym.make(id="Taxi-v3", max_episode_steps=hparams["max_steps"])
    test_env = gym.make(id="Taxi-v3", max_episode_steps=hparams["max_steps"])
    
    nnet = FCActionValue(
        in_dim=1,
        num_actions=int(train_env.action_space.n),
        h=[128],
        embed_in=train_env.observation_space.n,
        embed_dim=64
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
    EpsilonDecayHookFactory.initialize(hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)

    result = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=200,
        update_per_step=0.25, episode_per_test=100, batch_size=hparams["batch_size"],
        train_fn=EpsilonDecayHookFactory.train_hook,
        test_fn=lambda epoch, global_step: agent.set_eps(0.05),
        logger=logger
    ).run()