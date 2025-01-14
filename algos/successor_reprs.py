from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from gymnasium import Env
import numpy as np
from tqdm import tqdm
from envs.gridworld.gridworld import Gridworld
from tianshou.data.buffer.base import ReplayBuffer, Batch

# This file implements the code from:
# 1. Barreto, André, Will Dabney, Rémi Munos, Jonathan J. Hunt, Tom Schaul, Hado van Hasselt, and David Silver. “Successor Features for Transfer in Reinforcement Learning.” arXiv, April 12, 2018. http://arxiv.org/abs/1606.05312.
# 2. Machado, Marlos C, Andre Barreto, Doina Precup, and Michael Bowling. “Temporal Abstraction in Reinforcement Learning with the Successor Representation,” n.d.

class SRTabular:
    def __init__(self, n_states: int, n_acts: int, step_size: float, disc_fact: float, obs_to_ind: Callable, act_to_ind: Callable) -> None:
        
        # These functions are needed to convert observations and actions to their indexes in the Psi table. 
        # These functions should either be provided by the environment, or written with a specific environment in mind.
        self._obs_to_ind: Callable = obs_to_ind
        self._act_to_ind: Callable = act_to_ind

        self.n_states = n_states
        self.n_acts = n_acts
        self.step_size = step_size
        self.disc_fact = disc_fact

        # psi^\pi(s,a) from (Barreto et al., 2018) Eq. (4)
        # 1D tensor corresponding to all possible transitions \in R^{|S|^2|A|}
        self.psi_sparse = np.zeros(n_states**2 * n_acts)
        # 3D tensor counting all possible transitions. Compared to self.psi_sparse, trajectory-level information is kept here. 
        # Necessary for plotting successor representations.
        self.psi_dense = np.zeros((n_states, n_acts, n_states))
    
    # Psi from (Machado et al., 2021)
    @property
    def Psi(self):
        return np.sum(self.psi_dense, axis=1)

    def _transition_to_dense_index(self, obs: np.ndarray, action:np.ndarray, next_obs:np.ndarray) -> Tuple[int]:
        return self._obs_to_ind(obs), self._act_to_ind(action), self._obs_to_ind(next_obs)
    
    def _transition_to_one_hot_index(self, obs: np.ndarray, action:np.ndarray, next_obs:np.ndarray) -> int:
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
    
    def one_hot_index_to_vector(self, one_hot_index:int=None) -> np.ndarray:
        # This returns one-hot vector \phi(s,a,s') from (Barreto et al., 2018)
        one_hot_vec = np.zeros(self.n_states**2 * self.n_acts)
        if one_hot_index is not None:
            one_hot_vec[one_hot_index] = 1
        return one_hot_vec
    
    def select_action(self, obs: np.ndarray) -> int:
        return np.random.randint(0, self.n_acts)
    
    def store_weights(self, path:str) -> Tuple:
        np.savez(path, psi_sparse=self.psi_dense, psi_dense=self.psi_dense, Psi=self.Psi)
        return 'psi_sparse', 'psi_dense', 'Psi'
    
    @abstractmethod
    def store_trnsition(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: bool, truncated: bool) -> None:
        pass
    
    @abstractmethod
    def update(self) -> None:
        pass


class SROffPolicy(SRTabular):
    pass


class SROnPolicy(SRTabular):
    def __init__(self, n_states, n_acts, step_size, disc_fact, obs_to_ind, act_to_ind):
        super().__init__(n_states, n_acts, step_size, disc_fact, obs_to_ind, act_to_ind)
        self.trajectory = []
        self.done = False
        self.psis = []
    
    def store_trnsition(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: bool, truncated: bool) -> None:
        transition = Batch(obs=obs, act=action, obs_next=next_obs, rew=reward, terminated=terminated, truncated=truncated)
        self.trajectory.append(transition)
        if terminated or truncated:
            self.done = True
    
    def update(self) -> None:
        if self.done is False:
            return
        
        psi_trajectory = self.one_hot_index_to_vector(one_hot_index=None)
        for t in range(len(self.trajectory)-1, 0, -1):
            # Actions needed to update the sparse representation
            transition_t = self.trajectory[t]
            obs_t, action_t, next_obs_t = self._transition_to_dense_index(
                obs=transition_t.obs,
                action=transition_t.act,
                next_obs=transition_t.obs_next
            )
            one_hot_index_t = self._dense_index_to_one_hot_index(obs_ind=obs_t, action_ind=action_t, next_obs_ind=next_obs_t)
            phi_t = self.one_hot_index_to_vector(one_hot_index=one_hot_index_t)
            psi_trajectory += self.disc_fact**t * phi_t
            self.psis.append(psi_trajectory)
            psis = np.array(self.psis)
            self.psi_sparse = np.average(psis, axis=0)

            # Actions needed to update the dense representation
            for i in range(t, 0, -1):
                transition_i = self.trajectory[i]
                _, _, next_obs_i = self._transition_to_dense_index(
                    obs=transition_i.obs,
                    action=transition_i.act,
                    next_obs=transition_i.obs_next
                )
                self.psi_dense[obs_t, action_t, next_obs_i] += self.disc_fact**(t-i)

        # Reset trajectory once update is completed
        self.trajectory = []
        self.done = False


def train_loop(env: Env, agent: SRTabular, hparams: Dict) -> None:
    for _ in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_trnsition(
                obs=obs, 
                next_obs=next_obs,
                action=action, 
                reward=reward, 
                terminated=terminated, 
                truncated=truncated
            )
            if hparams['algo_type'] == 'off_policy':
                agent.update(batch_size=hparams['batch_size'])
            
            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
        
        # On-policy psi update
        if hparams['algo_type'] == 'on_policy':
            agent.update()

def plot_Psi(env: Env, weights_path:str, weights_keys: Tuple, states: Dict, name_prefix:str) -> None:
    psi_tables = np.load(weights_path)
    Psi = psi_tables[weights_keys[2]]

    for state_id, state_obs in states.items():
        psi_s = Psi[env.obs_to_id(observation=state_obs), :]
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
        agent = SROnPolicy(
            n_states=env.n_states, 
            n_acts=env.action_space.n,
            step_size=hparams['step_size'], 
            disc_fact=hparams['disc_fact'],
            obs_to_ind=env.obs_to_id,
            act_to_ind=env.act_to_id
        )
    elif hparams['algo_type'] == 'off_policy':
        agent = SROffPolicy(
            n_states=env.n_states, 
            n_acts=env.action_space.n,
            rb=rb, 
            step_size=hparams['step_size'], 
            disc_fact=hparams['disc_fact'],
            obs_to_ind=env.obs_to_id,
            act_to_ind=env.act_to_id
        )
    else:
        raise ValueError('Parameter algo_type not supported!')
   
    _ = train_loop(
        env=env,
        agent=agent,
        hparams=hparams
    )
    weights_path = os.path.join(store_path, f'{hparams["algo_type"]}_psi.npz')
    weights_keys = agent.store_weights(path=weights_path)
    
    plot_Psi(
        env=env,
        weights_path=weights_path, 
        weights_keys=weights_keys,
        states=hparams['states'], 
        name_prefix=f'{hparams["algo_type"]}_state'
    )
    