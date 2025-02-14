from abc import abstractmethod
from typing import Union, Callable, Tuple, Dict
from tianshou.data.batch import Batch
from gymnasium.spaces import Space
from gymnasium import Env
import numpy as np
import torch
from tqdm import tqdm
from envs.gridworld.gridworld import Gridworld
from common import TabularAgent, train_loop, Scheduler
from tianshou.data.buffer.base import ReplayBuffer, Batch


class QLearningTabular(TabularAgent):
    def __init__(self, params:Dict, rb: ReplayBuffer, obs_to_inds: Callable, acts_to_inds: Callable, epsilon_scheduler: Scheduler) -> None:
        super().__init__(params=params)

        self.step_size = params["step_size"]
        self.disc_fact = params["disc_fact"]
        self.obs_to_inds = obs_to_inds
        self.acts_to_inds = acts_to_inds

        self.rb: ReplayBuffer = rb
        self.warmup_steps = params["warmup_steps"]
        self.total_steps = 0
        
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = None

        # Q-table initialized with zeros
        self.q_table = torch.zeros((self.n_states, self.n_acts))

        # Seeding random generators for reproducibility
        if params["seed"] is not None:
            self.rng = torch.Generator().manual_seed(params["seed"])
        else:
            self.rng = torch.Generator()
    
    def store_transition(self, obs:Union[torch.Tensor, np.ndarray], next_obs:Union[torch.Tensor, np.ndarray], action:Union[torch.Tensor, np.ndarray], reward:Union[torch.Tensor, np.ndarray], terminated: bool, truncated: bool) -> None:
        obs, next_obs, action, reward, terminated, truncated = super().store_transition(obs, next_obs, action, reward, terminated, truncated)
        transition = Batch(obs=obs, act=action, obs_next=next_obs, rew=reward, terminated=terminated, truncated=truncated)
        self.rb.add(transition)

    def select_action(self, obs: torch.Tensor) -> int:
        """Epsilon-greedy action selection."""
        self.epsilon = self.epsilon_scheduler.step()
        if torch.rand(1, generator=self.rng).item() < self.epsilon:
            action = torch.randint(low=0, high=self.n_acts, size=(1,))
        else:
            obs_ind = self.obs_to_inds(obs)
            action = torch.argmax(self.q_table[obs_ind])
        
        self.total_steps += 1
        return action

    def update(self, batch_size:int=0) -> None:
        batch, _ = self.rb.sample(batch_size=batch_size)
        batch.to_torch()

        if len(batch) < 1 or self.total_steps < self.warmup_steps:
            return
        
        obs = batch.obs
        obs_inds = self.obs_to_inds(obs)
        acts = batch.act
        act_inds = self.acts_to_inds(acts)
        next_obs = batch.obs_next
        next_obs_inds = self.obs_to_inds(next_obs)
        rewards = batch.rew
        terminated = batch.terminated

        q_currents = self.q_table[obs_inds, act_inds]
        q_nexts = torch.max(self.q_table[next_obs_inds], dim=1).values
        q_nexts = q_nexts * (1 - terminated.float())  # Zero out Q_next if terminated
        td_targets = rewards + self.disc_fact * q_nexts
        td_errors = td_targets - q_currents
        self.q_table[obs_inds, act_inds] += self.step_size * td_errors
    
    def store_weights(self, path: str) -> Tuple:
        torch.save({'q_table': self.q_table}, path)
        return ('q_table', )
    

class MonteCarloTabular(TabularAgent):
    def __init__(self, params:Dict, obs_to_inds: Callable, acts_to_inds: Callable, epsilon_scheduler: Scheduler) -> None:
        super().__init__(params=params)

        self.disc_fact = params["disc_fact"]
        self.obs_to_inds = obs_to_inds
        self.acts_to_inds = acts_to_inds
        
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = None

        # Value table and visit counts
        self.q_table = torch.zeros((self.n_states, self.n_acts))
        self.returns_sum = torch.zeros((self.n_states, self.n_acts))
        self.returns_count = torch.zeros((self.n_states, self.n_acts))

        # To store the current episode trajectory
        self.trajectory = []

        # Seeding random generators for reproducibility
        if params["seed"] is not None:
            self.rng = torch.Generator().manual_seed(params["seed"])
        else:
            self.rng = torch.Generator()

    def select_action(self, obs: torch.Tensor) -> int:
        """Epsilon-greedy action selection."""
        self.epsilon = self.epsilon_scheduler.step()
        if torch.rand(1, generator=self.rng).item() < self.epsilon:
            action = torch.randint(low=0, high=self.n_acts, size=(1,))
        else:
            obs_ind = self.obs_to_inds(obs)
            action = torch.argmax(self.q_table[obs_ind])
        return action

    def store_transition(self, obs:Union[torch.Tensor, np.ndarray], next_obs:Union[torch.Tensor, np.ndarray], action:Union[torch.Tensor, np.ndarray], reward:Union[torch.Tensor, np.ndarray], terminated: bool, truncated: bool) -> None:
        self.trajectory.append((obs, action, reward))

    def update(self) -> None:
        """Perform first-visit MC update after an episode ends."""
        G = 0
        visited = set()

        for obs, action, reward in reversed(self.trajectory):
            obs_ind = self.obs_to_inds(obs)
            act_ind = self.acts_to_inds(action)
            
            G = self.disc_fact * G + reward
            if (obs_ind.item(), action.item()) not in visited:
                visited.add((obs_ind.item(), act_ind.item()))
                self.returns_sum[obs_ind, act_ind] += G
                self.returns_count[obs_ind, act_ind] += 1
                self.q_table[obs_ind, act_ind] = self.returns_sum[obs_ind, act_ind] / self.returns_count[obs_ind, act_ind]

        # Clear the trajectory after updating
        self.trajectory = []
    
    def store_weights(self, path: str) -> Tuple:
        torch.save({'q_table': self.q_table}, path)
        return ('q_table', )


def plot_V(env: Env, weights_path: str, weights_keys:Tuple, name_prefix: str) -> None:
    q_table = torch.load(weights_path, weights_only=False)[weights_keys[0]]
    v_table = torch.mean(q_table, axis=1).reshape(env.grid_shape)
    env.plot_values(table=v_table, plot_name=name_prefix)


if __name__ == '__main__':
    import os
    from utils import setup_artefact_paths

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

    hparams["n_states"] = env.n_states
    hparams["n_acts"] = env.action_space.n

    if hparams['algo_type'] == 'on_policy':
        agent = MonteCarloTabular(
            params=hparams,
            obs_to_inds=env.obs_to_ids,
            acts_to_inds=env.acts_to_ids,
            epsilon_scheduler=scheduler
        )
    elif hparams['algo_type'] == 'off_policy':
        agent = QLearningTabular(
            params=hparams,
            rb=rb,
            obs_to_inds=env.obs_to_ids,
            acts_to_inds=env.acts_to_ids,
            epsilon_scheduler=scheduler
        )
    else:
        raise ValueError('Parameter algo_type not supported!')
   
    _ = train_loop(
        env=env,
        agent=agent,
        hparams=hparams,
        random=False
    )
    weights_path = os.path.join(store_path, f'{hparams["algo_type"]}_qtable.pt')
    weights_keys = agent.store_weights(path=weights_path)
    
    plot_V(
        env=env,
        weights_path=weights_path, 
        weights_keys=weights_keys, 
        name_prefix=f'{hparams["algo_type"]}_vtable'
    )