import os
from typing import Dict
import torch
import math

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.nets import ConcatActionValue, precompute_instruction_embeddings
from algos.common import EpsilonDecayHookFactory, SaveHookFactory
from envs.taxicab.language_taxicab import LanguageTaxicab, LanguageTaxicabFactory
from utils import setup_experiment


if __name__ == '__main__':
    
    script_path = os.path.abspath(__file__)
    experiment_name, store_path, yaml_path, precomp_path = setup_experiment(script_path=script_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
    writer = SummaryWriter(store_path)
    logger = TensorboardLogger(writer)

    env_factory = LanguageTaxicabFactory(
        hparams=hparams, 
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
        h=hparams["hidden_dim"],
        precom_embeddings=precomp_embeddings,
        device=device
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