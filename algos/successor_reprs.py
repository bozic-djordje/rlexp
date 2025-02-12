from abc import abstractmethod
from collections import defaultdict
from typing import Union, Callable, Dict, Tuple
from gymnasium import Env
import numpy as np
import torch
from tianshou.data.buffer.base import ReplayBuffer, Batch
from envs.gridworld.gridworld import Gridworld
from common import Agent, train_loop

# This file implements the code from:
# 1. Barreto, André, Will Dabney, Rémi Munos, Jonathan J. Hunt, Tom Schaul, Hado van Hasselt, and David Silver. “Successor Features for Transfer in Reinforcement Learning.” arXiv, April 12, 2018. http://arxiv.org/abs/1606.05312.
# 2. Machado, Marlos C, Andre Barreto, Doina Precup, and Michael Bowling. “Temporal Abstraction in Reinforcement Learning with the Successor Representation,” n.d.

class SFTabular(Agent):
    def __init__(self, n_states: int, n_acts: int, step_size: float, disc_fact: float, obs_to_inds: Callable, acts_to_inds: Callable) -> None:
        super().__init__()
        self._obs_to_inds: Callable = obs_to_inds
        self._acts_to_inds: Callable = acts_to_inds

        self.n_states = n_states
        self.n_acts = n_acts
        self.step_size = step_size
        self.disc_fact = disc_fact

        self.psi_sparse = torch.zeros((n_states, n_acts, n_states**2 * n_acts))
        self.psi_dense = torch.zeros((n_states, n_acts, n_states))

    @property
    def Psi(self):
        return torch.sum(self.psi_dense, dim=1)

    def _transition_to_dense_index(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> Tuple[int]:
        return self._obs_to_inds(obs).item(), self._acts_to_inds(int(action)).item(), self._obs_to_inds(next_obs).item()
    
    def _transition_to_one_hot_index(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> int:
        obs_ind, action_ind, next_obs_ind = self._transition_to_dense_index(obs=obs, action=action, next_obs=next_obs)
        return self._dense_index_to_one_hot_index(obs_ind=obs_ind, next_obs_ind=next_obs_ind, action_ind=action_ind)

    def _dense_index_to_one_hot_index(self, obs_ind: int, action_ind: int, next_obs_ind: int) -> int:
        index = obs_ind * self.n_acts * self.n_states + action_ind * self.n_states + next_obs_ind
        return index
    
    def _one_hot_index_to_dense_index(self, one_hot_index: int) -> int:
        obs_ind = one_hot_index // (self.n_acts * self.n_states)
        remaining = one_hot_index % (self.n_acts * self.n_states)
        act_ind = remaining // self.n_states
        next_obs_ind = remaining % self.n_states
        return obs_ind, act_ind, next_obs_ind
    
    def one_hot_index_to_vector(self, one_hot_index: int = None) -> torch.Tensor:
        one_hot_vec = torch.zeros(self.n_states**2 * self.n_acts)
        if one_hot_index is not None:
            one_hot_vec[one_hot_index] = 1
        return one_hot_vec
    
    def select_action(self, obs: torch.Tensor) -> int:
        return torch.randint(0, self.n_acts, (1,)).item()
    
    def store_weights(self, path: str) -> Tuple:
        torch.save({'psi_sparse': self.psi_sparse, 'psi_dense': self.psi_dense, 'Psi': self.Psi}, path)
        return 'psi_sparse', 'psi_dense', 'Psi'


class SFOffPolicy(SFTabular):
    def __init__(self, n_states, n_acts, rb: ReplayBuffer, step_size, disc_fact, obs_to_ind, act_to_ind):
        super().__init__(n_states, n_acts, step_size, disc_fact, obs_to_ind, act_to_ind)
        self.rb: ReplayBuffer = rb
        self.trajectory = []
        self.done = False
    
    def store_transition(self, obs:Union[torch.Tensor, np.ndarray], next_obs:Union[torch.Tensor, np.ndarray], action:Union[torch.Tensor, np.ndarray], reward:Union[torch.Tensor, np.ndarray], terminated: bool, truncated: bool) -> None:
        obs, next_obs, action, reward, terminated, truncated = super().store_transition(obs, next_obs, action, reward, terminated, truncated)
        transition = Batch(obs=obs, act=action, obs_next=next_obs, rew=reward, terminated=terminated, truncated=truncated)
        self.rb.add(transition)
    
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
            one_hot_index = self._dense_index_to_one_hot_index(obs_ind=obs_ind, action_ind=action_ind, next_obs_ind=next_obs_ind)
            phi_t = self.one_hot_index_to_vector(one_hot_index=one_hot_index)
            expect_pi = torch.mean(self.psi_sparse[next_obs_ind, :, :], dim=0)
            self.psi_sparse[obs_ind, action_ind, :] = phi_t + self.disc_fact * expect_pi

            for i in range(self.n_states):
                td_error = 1 if obs_ind == i else 0
                td_error += self.disc_fact * torch.mean(self.psi_dense[next_obs_ind, :, i]) - self.psi_dense[obs_ind, action_ind, i]
                self.psi_dense[obs_ind, action_ind, i] += self.step_size * td_error
                
            i += 1
            if transition.terminated is True:
                i += 1


class SFOnPolicy(SFTabular):
    def __init__(self, n_states, n_acts, step_size, disc_fact, obs_to_ind, act_to_ind):
        super().__init__(n_states, n_acts, step_size, disc_fact, obs_to_ind, act_to_ind)
        self.trajectory = []
        self.done = False
        self.psi_sa_rollouts = defaultdict(list)
    
    def store_transition(self, obs:Union[torch.Tensor, np.ndarray], next_obs:Union[torch.Tensor, np.ndarray], action:Union[torch.Tensor, np.ndarray], reward:Union[torch.Tensor, np.ndarray], terminated: bool, truncated: bool) -> None:
        obs, next_obs, action, reward, terminated, truncated = super().store_transition(obs, next_obs, action, reward, terminated, truncated)
        transition = Batch(obs=obs, act=action, obs_next=next_obs, rew=reward, terminated=terminated, truncated=truncated)
        self.trajectory.append(transition)
        if terminated or truncated:
            self.done = True
    
    def update(self) -> None:
        if self.done is False:
            return
        
        for t in range(len(self.trajectory) - 1, 0, -1):
            transition_t = self.trajectory[t]
            obs_t, action_t, next_obs_t = self._transition_to_dense_index(
                obs=transition_t.obs,
                action=transition_t.act,
                next_obs=transition_t.obs_next
            )

            psi_sa = self.one_hot_index_to_vector(one_hot_index=None)
            # See (Barreto et al., 2018) Equation (3). For i=t to end of trajectory we compute the sum 
            # over discounted \phi_i and store it into psi_sa.
            for i in range(t, len(self.trajectory)):
                transition_i = self.trajectory[i]
                obs_i, action_i, next_obs_i = self._transition_to_dense_index(
                    obs=transition_i.obs,
                    action=transition_i.act,
                    next_obs=transition_i.obs_next
                )

                # See (Barreto et al., 2018) Equation (3).
                one_hot_index_i = self._dense_index_to_one_hot_index(obs_ind=obs_i, action_ind=action_i, next_obs_ind=next_obs_i)
                phi_i = self.one_hot_index_to_vector(one_hot_index=one_hot_index_i)
                psi_sa += self.disc_fact**(i - t) * phi_i
                
            self.psi_sa_rollouts[(obs_t, action_t)].append(psi_sa.tolist())
            psis = torch.tensor(self.psi_sa_rollouts[(obs_t, action_t)])
            self.psi_sparse[obs_t, action_t, :] = torch.mean(psis, dim=0)

            # Actions needed to update the dense representation. See your notebook.
            for i in range(t, 0, -1):
                transition_i = self.trajectory[i]
                _, _, next_obs_i = self._transition_to_dense_index(
                    obs=transition_i.obs,
                    action=transition_i.act,
                    next_obs=transition_i.obs_next
                )
                self.psi_dense[obs_t, action_t, next_obs_i] += self.disc_fact**(t - i)

        # Reset trajectory once update is completed
        self.trajectory = []
        self.done = False


def plot_Psi(env: Env, weights_path: str, weights_keys:Tuple, states: Dict, name_prefix: str) -> None:
    psi_tables = torch.load(weights_path)
    Psi = psi_tables[weights_keys[2]]

    for state_id, state_obs in states.items():
        psi_s = Psi[env.obs_to_ids(observations=np.array(state_obs)), :]
        psi_s = psi_s.reshape(env.grid_shape)
        env.plot_values(table=psi_s, plot_name=f'{name_prefix}_{state_id}')


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
        seed=hparams['seed']
    )

    rb = ReplayBuffer(size=hparams['buffer_size'])
    
    if hparams['algo_type'] == 'on_policy':
        agent = SFOnPolicy(
            n_states=env.n_states, 
            n_acts=env.action_space.n,
            step_size=hparams['step_size'], 
            disc_fact=hparams['disc_fact'],
            obs_to_ind=env.obs_to_ids,
            act_to_ind=env.acts_to_ids
        )
    elif hparams['algo_type'] == 'off_policy':
        agent = SFOffPolicy(
            n_states=env.n_states, 
            n_acts=env.action_space.n,
            rb=rb, 
            step_size=hparams['step_size'], 
            disc_fact=hparams['disc_fact'],
            obs_to_ind=env.obs_to_ids,
            act_to_ind=env.acts_to_ids
        )
    else:
        raise ValueError('Parameter algo_type not supported!')
   
    _ = train_loop(
        env=env,
        agent=agent,
        hparams=hparams
    )
    weights_path = os.path.join(store_path, f'{hparams["algo_type"]}_psi.pt')
    weights_keys = agent.store_weights(path=weights_path)
    
    plot_Psi(
        env=env,
        weights_path=weights_path, 
        weights_keys=weights_keys,
        states=hparams['states'], 
        name_prefix=f'{hparams["algo_type"]}_state'
    )