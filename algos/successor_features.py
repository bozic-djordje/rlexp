from abc import abstractmethod
from collections import defaultdict
from typing import Union, Callable, Dict, Tuple
from gymnasium import Env, Space
import numpy as np
import torch
from torch import nn
from tianshou.data.buffer.base import ReplayBuffer, Batch
from envs.gridworld.gridworld import Gridworld
from common import Agent, Scheduler, train_loop
from successor_reprs import SFTabular
import matplotlib.pyplot as plt


class SFQLearning(SFTabular):
    def __init__(self, n_states, action_space, rb: ReplayBuffer, step_size, disc_fact, obs_to_ind, act_to_ind, device, epsilon_scheduler, warmup_steps, d):
        super().__init__(n_states, action_space.n, step_size, disc_fact, obs_to_ind, act_to_ind, device, d=d)
        self.rb: ReplayBuffer = rb
        self.action_space: Space = action_space
        # Initialise weights to some small random values
        self.w = torch.rand(size=(self.d, 1), requires_grad=True, device=self.device)
        self.w.data *= 0.001
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.SGD([self.w], lr=step_size)
        self.loss_history = []
        self.done = False

        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = None
    
    @property
    def q_table(self):
        return torch.matmul(self.psi_sparse, self.w).detach()
    
    def store_transition(self, obs:Union[torch.Tensor, np.ndarray], next_obs:Union[torch.Tensor, np.ndarray], action:Union[torch.Tensor, np.ndarray], reward:Union[torch.Tensor, np.ndarray], terminated: bool, truncated: bool) -> None:
        obs, next_obs, action, reward, terminated, truncated = super().store_transition(obs, next_obs, action, reward, terminated, truncated)
        transition = Batch(obs=obs, act=action, obs_next=next_obs, rew=reward, terminated=terminated, truncated=truncated)
        self.rb.add(transition)

    def select_action(self, obs: torch.Tensor, update:bool=True) -> int:
        """Epsilon-greedy action selection."""
        if update:
            self.epsilon = self.epsilon_scheduler.step()
        
        if torch.rand(1).item() < self.epsilon:
            act_ind = torch.tensor(self.action_space.sample())
        else:
            q_sa = torch.empty(self.n_acts)
            obs_ind = self._obs_to_inds(obs)
            # Vectorized Q-value computation: shape (A, d) @ (d,)
            q_sa = torch.matmul(self.psi_sparse[obs_ind], self.w)  # Shape (A,)
            # Select the action with the highest Q-value
            act_ind = torch.argmax(q_sa).item()
        
        if update:
            self.total_steps += 1
        
        return act_ind
    
    def update(self, batch_size:int=0):
        batch, _ = self.rb.sample(batch_size=batch_size)
        if len(batch) < 1:
            return
        
        i = 0
        while i < len(batch):
            transition = batch[i]
            obs_ind, action_ind, next_obs_ind = self._transition_to_dense_index(
                obs=transition.obs,
                action=transition.act,
                next_obs=transition.obs_next
            )

            # Update (Barreto et al., 2018) Equation (3).
            one_hot_index = self._dense_index_to_one_hot_index(obs_ind=obs_ind, action_ind=action_ind, next_obs_ind=next_obs_ind)
            phi_t = self.one_hot_index_to_vector(one_hot_index=one_hot_index)
            # TODO: See why this doesn't propagate correctly. You have already had the same error!
            next_action = self.select_action(obs=transition.obs_next, update=False)
            next_action_ind = self._acts_to_inds(next_action)
            expect_pi = self.psi_sparse[next_obs_ind, next_action_ind, :]
            self.psi_sparse[obs_ind, action_ind, :] = phi_t + self.disc_fact * expect_pi

            # Update self.w in a supervised fashion
            # Forward pass
            r_pred = torch.matmul(phi_t, self.w)
            r_target = torch.unsqueeze(torch.tensor(transition.rew, dtype=torch.float32, device=self.device), dim=0)
            loss = self.loss_fn(r_pred, r_target)
            self.loss_history.append(loss.item())
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
                
            i += 1
            if transition.terminated is True:
                i += 1
    
    def store_weights(self, path: str) -> Tuple:
        torch.save({'psi_sparse': self.psi_sparse.cpu(), 'q_table': self.q_table.cpu()}, path)
        return 'psi_sparse', 'q_table'


def plot_V(env: Env, weights_path: str, weights_keys:Tuple, name_prefix: str) -> None:
    q_table = torch.load(weights_path, weights_only=False)[weights_keys[1]]
    v_table = torch.mean(q_table, axis=1).reshape(env.grid_shape)
    env.plot_values(table=v_table, plot_name=name_prefix)


def plot_l2_loss(loss_values, save_path):
    """
    Plots a list of L2 loss values and saves the figure to a specified path.

    Parameters:
    - loss_values (list of float): A list of L2 loss values.
    - save_path (str): The path where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_values, label="L2 Loss", color="blue", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("L2 Loss")
    plt.title("L2 Loss Over Iterations")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    import os
    from utils import setup_artefact_paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path)
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
    env = Gridworld(
        grid=hparams['grid'],
        store_path=store_path, 
        max_steps=hparams['max_steps'], 
        seed=hparams['seed'],
        start_pos=hparams['start_pos']
    )

    rb = ReplayBuffer(size=hparams['buffer_size'])
    scheduler = Scheduler(start=hparams['schedule_start'], end=hparams['schedule_end'], decay_func=lambda step: hparams['schedule_decay'] * step)
    
    agent = SFQLearning(
        n_states=env.n_states, 
        action_space=env.action_space,
        rb=rb, 
        step_size=hparams['step_size'], 
        disc_fact=hparams['disc_fact'],
        obs_to_ind=env.obs_to_ids,
        act_to_ind=env.acts_to_ids,
        device=device,
        epsilon_scheduler=scheduler,
        warmup_steps=hparams['warmup_steps'],
        d=hparams['hidden_dim']
    )
   
    _ = train_loop(
        env=env,
        agent=agent,
        hparams=hparams,
        random=False
    )
    weights_path = os.path.join(store_path, f'{hparams["algo_type"]}_psi.pt')
    weights_keys = agent.store_weights(path=weights_path)
    
    plot_V(
        env=env,
        weights_path=weights_path, 
        weights_keys=weights_keys,
        name_prefix=f'{hparams["algo_type"]}_vtable'
    )
    
    loss_path = os.path.join(store_path, f'{hparams["algo_type"]}_loss')
    plot_l2_loss(
        loss_values=agent.loss_history, 
        save_path=loss_path
    )