from abc import abstractmethod, ABC
from tianshou.trainer import BaseTrainer
from tianshou.data.collector import CollectStatsBase
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats
from tianshou.policy import BasePolicy
from tianshou.utils import TensorboardLogger
from tianshou.data import PrioritizedReplayBuffer, ReplayBuffer, Batch
from collections import defaultdict
from gymnasium import Env
from tqdm import tqdm
from typing import Any, Dict, Callable, List, Tuple, Union
import torch
import numpy as np
from collections import defaultdict
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


class EpochHook(ABC):
    def __init__(self, agent: BasePolicy, logger: TensorboardLogger):
        self.agent = agent
        self.logger = logger

    @abstractmethod
    def hook(self, epoch: int, global_step: int):
        """Hook function to be implemented by subclasses."""
        pass


class CompositeHook(EpochHook):
    def __init__(self, hooks:List, agent: BasePolicy, logger: TensorboardLogger):
        super().__init__(agent, logger)
        self.hooks: List[EpochHook] = hooks

    def add_hook(self, hook:EpochHook):
        self.hooks.append(hook)
    
    def hook(self, epoch, global_step):
        for h in self.hooks:
            h.hook(epoch=epoch, global_step=global_step)


class EpsilonDecayHook(EpochHook):
    def __init__(self, hparams: Dict, max_steps: int, agent: BasePolicy, logger: TensorboardLogger, is_linear=False):
        super().__init__(agent, logger)

        self.eps_start = hparams["epsilon_start"]
        self.eps_end = hparams["epsilon_end"]
        decay_fraction = hparams["epsilon_fraction"]
        self.decay_steps = int(max_steps * decay_fraction)
        self.is_linear = is_linear

        if not self.is_linear:
            delta = 1e-3
            self.k = -math.log(delta / (self.eps_start - self.eps_end)) / self.decay_steps

    def hook(self, epoch: int, global_step: int):
        if global_step <= self.decay_steps:
            if self.is_linear:
                epsilon = self.eps_start - (self.eps_start - self.eps_end) * (global_step / self.decay_steps)
            else:
                epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.k * global_step)
        else:
            epsilon = self.eps_end

        self.agent.set_eps(epsilon)
        self.logger.write("train/env_step", global_step, {"epsilon": self.agent.eps})


class BetaAnnealHook(EpochHook):
    def __init__(self, agent: BasePolicy, buffer:PrioritizedReplayBuffer, beta_start:float, beta_end:float, frac:float, max_steps:int, logger: TensorboardLogger):
        super().__init__(agent, logger)
        self.rb = buffer
        self.b0 = beta_start
        self.b1 = beta_end
        self.anneal_steps = int(frac * max_steps)

    def hook(self, epoch: int, global_step: int):
        if global_step >= self.anneal_steps:
            beta = self.b1
        else:
            beta = self.b0 + (self.b1 - self.b0) * (global_step / self.anneal_steps)
        self.rb.set_beta(beta)


class SaveHook:
    def __init__(self, save_path:str):
        self.save_path = save_path

    @abstractmethod
    def hook(self, agent: BasePolicy):
        torch.save(agent.state_dict(), self.save_path)

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


def argmax_random_tiebreak(qs: torch.Tensor) -> torch.Tensor:
    """
    qs: Tensor of shape (batch_size, num_actions)
    Returns: LongTensor of shape (batch_size,) with random argmax indices
    """
    max_vals = qs.max(dim=1, keepdim=True).values               # (B, 1)
    is_max = qs == max_vals                                     # (B, A), bool mask
    rand = torch.rand_like(qs)                                  # (B, A), uniform random
    rand[~is_max] = -1.0                                         # mask out non-max entries
    return rand.argmax(dim=1)                                   # pick random max index


class SFTrainer(BaseTrainer):
    """Off-policy trainer, but unlike Tianshou's off policy trainer, this one passes the entire buffer to .update.
    Note that it is expected that the learn method of a policy will perform
    batching when using this trainer.
    """
    
    def policy_update_fn(
        self, 
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Perform `update_per_step * n_collected_steps` gradient steps by sampling mini-batches from the buffer.

        :param collect_stats: the :class:`~TrainingStats` instance returned by the last gradient step. Some values
            in it will be replaced by their moving averages.
        """
        assert self.train_collector is not None
        n_collected_steps = collect_stats.n_collected_steps
        n_gradient_steps = round(self.update_per_step * n_collected_steps)
        if n_gradient_steps == 0:
            raise ValueError(
                f"n_gradient_steps is 0, n_collected_steps={n_collected_steps}, "
                f"update_per_step={self.update_per_step}",
            )
        for _ in range(n_gradient_steps):
            update_stat = self.policy.update(
                sample_size=self.batch_size,
                buffer=self.train_collector.buffer
            )

            # logging
            self.policy_update_time += update_stat.train_time
        # Only the last update stat returned, like in Tianshou's version
        return update_stat


class GroupedReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, **kwargs):
        super().__init__(size, **kwargs)

        # Maps instr -> list of indices in the buffer
        self.group_index: Dict[str, List[int]] = defaultdict(list)

        # Reverse mapping: buffer index -> instr
        self.index_to_instr: np.ndarray = np.full(size, None, dtype=object)

    def add(self, batch: Union[Batch, Dict[str, Any]], buffer_ids: Any = None) -> List[int]:
        
        ret = super().add(batch, buffer_ids)
        if isinstance(ret, tuple) and len(ret) == 4:
            indices, ep_rew, ep_len, ep_idx = ret
            return_tuple = (ep_rew, ep_len, ep_idx)
        else:
            indices = ret
            return_tuple = ()

        instrs = batch.obs.instr
        if isinstance(instrs, (int, str)):
            instrs = [instrs]

        for i, idx in enumerate(indices):
            new_instr = instrs[i]

            # Remove the index from the old instr's list (if any)
            old_instr = self.index_to_instr[idx]
            if old_instr is not None:
                try:
                    self.group_index[old_instr].remove(idx)
                    if not self.group_index[old_instr]:
                        del self.group_index[old_instr]  # clean up empty lists
                except ValueError:
                    pass

            # Add the new instr -> idx mapping
            self.group_index[new_instr].append(idx)
            self.index_to_instr[idx] = new_instr

        if return_tuple:
            return (indices, *return_tuple)
        else:
            return indices
    
    def sample_group_id(self, sample_size:int) -> Any:
        if len(self.group_index) == 0:
            raise ValueError(f"No samples in the replay buffer.")
        eligible_keys = [k for k, idxs in self.group_index.items() if len(idxs) > sample_size]
        return np.random.choice(eligible_keys)
    
    def total_group_id(self, group_id:str) -> int:
        if group_id in self.group_index:
            return len(self.group_index[group_id])
        else:
            return 0

    def sample_group(self, group_id: Any, batch_size: int) -> Batch:
        """Sample transitions with obs['instr'] == instr."""
        indices = self.group_index.get(group_id, [])
        if len(indices) < batch_size:
            raise ValueError(f"Not enough samples for instr={group_id}: only {len(indices)} available.")
        chosen = np.random.choice(indices, size=batch_size, replace=False)
        return self[chosen], indices

    def clear(self) -> None:
        super().clear()
        self.group_index.clear()
        self.index_to_instr[:] = None
