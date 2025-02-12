from abc import abstractmethod
from gymnasium import Env
from tqdm import tqdm
from typing import Dict
import torch
import numpy as np
from typing import Union, Tuple

# TODO: Use this class to define some Agent properties, now useful for type hinting
class Agent:

    def store_transition(self, obs:Union[torch.Tensor, np.ndarray, np.generic], next_obs:Union[torch.Tensor, np.ndarray, np.generic], action:Union[torch.Tensor, np.ndarray, np.generic], reward:Union[torch.Tensor, np.ndarray, np.generic], terminated: bool, truncated: bool) -> Tuple:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        elif isinstance(obs, np.generic):
            obs = torch.tensor(obs.item())

        if isinstance(next_obs, np.ndarray):
            next_obs = torch.from_numpy(next_obs)
        elif isinstance(next_obs, np.generic):
            next_obs = torch.tensor(next_obs.item())

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        elif isinstance(action, np.generic):
            action = torch.tensor(action.item())

        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)
        elif isinstance(reward, np.generic):
            reward = torch.tensor(reward.item())
        
        return obs, next_obs, action, reward, terminated, truncated

    @abstractmethod
    def update():
        pass

    @abstractmethod
    def store_weights():
        pass


def train_loop(env: Env, agent: Agent, hparams: Dict) -> None:
    for _ in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(
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