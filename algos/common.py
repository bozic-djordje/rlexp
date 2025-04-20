from abc import abstractmethod, ABC
from tianshou.policy import BasePolicy
from tianshou.utils import TensorboardLogger
from collections import defaultdict
from gymnasium import Env
from tqdm import tqdm
from typing import Any, Dict, Callable
import torch
import numpy as np
from typing import Union, Tuple
import math


# TODO: Use this class to define some Agent properties, now useful for type hinting
class Agent:
    def __init__(self, params:Dict):
        self.history = defaultdict(list)

    def store_transition(self, obs:Union[torch.Tensor, np.ndarray, np.generic], next_obs:Union[torch.Tensor, np.ndarray, np.generic], action:Union[torch.Tensor, np.ndarray, np.generic], reward:Union[torch.Tensor, np.ndarray, np.generic], terminated: bool, truncated: bool) -> Tuple:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        elif isinstance(obs, np.generic):
            obs = torch.tensor(obs.item())
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)

        if isinstance(next_obs, np.ndarray):
            next_obs = torch.from_numpy(next_obs)
        elif isinstance(next_obs, np.generic):
            next_obs = torch.tensor(next_obs.item())
        elif not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(next_obs)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        elif isinstance(action, np.generic):
            action = torch.tensor(action.item())
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        
        return obs, next_obs, action, reward, terminated, truncated

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def store_weights(self):
        pass

    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def select_action(self, obs: torch.Tensor) -> Any:
        pass


class TabularAgent(Agent):
    def __init__(self, params: Dict):
        super().__init__(params=params)
        self.n_states = params["n_states"]
        self.n_acts = params["n_acts"]
    
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
        one_hot_vec = torch.zeros(self.d, device=self.device)
        if one_hot_index is not None:
            one_hot_vec[one_hot_index] = 1
        return one_hot_vec


class Scheduler:
    def __init__(self, start: float, end: float, decay_func: Callable):
        self.start = start
        self.end = end
        self.current = start
        self.decay_func = decay_func

    def step(self):
        self.current = max(self.end, self.decay_func(self.current))
        return self.current


class HookFactory(ABC):
    def __init__(self, agent: BasePolicy, logger: TensorboardLogger):
        self.agent = agent
        self.logger = logger

    @abstractmethod
    def hook(self, epoch: int, global_step: int):
        """Hook function to be implemented by subclasses."""
        pass


class EpsilonDecayHookFactory(HookFactory):
    def __init__(self, hparams: Dict, max_steps: int, agent: BasePolicy, logger: TensorboardLogger):
        super().__init__(agent, logger)

        self.eps_start = hparams["schedule_start"]
        self.eps_end = hparams["schedule_end"]
        decay_fraction = hparams["schedule_fraction"]
        self.decay_steps = int(max_steps * decay_fraction)

        delta = 1e-3
        self.k = -math.log(delta / (self.eps_start - self.eps_end)) / self.decay_steps

    def hook(self, epoch: int, global_step: int):
        if global_step <= self.decay_steps:
            epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.k * global_step)
        else:
            epsilon = self.eps_end

        self.agent.set_eps(epsilon)
        self.logger.write("train/env_step", global_step, {"epsilon": self.agent.eps})


#TODO: Remove stochastic once SFeatures become real agents
def train_loop(env: Env, agent: Agent, hparams: Dict, random=False) -> None:
    for _ in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset()
        done = False

        while not done:
            
            action = env.action_space.sample() if random else agent.select_action(obs=obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(
                obs=obs, 
                next_obs=next_obs,
                action=action, 
                reward=reward, 
                terminated=terminated, 
                truncated=truncated
            )
            if hparams['algo_type'] == 'off_policy' or hparams['algo_type'] == 'sfqlearning':
                agent.update(batch_size=hparams['batch_size'])
            
            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
        
        # On-policy psi update
        if hparams['algo_type'] == 'on_policy':
            agent.update()