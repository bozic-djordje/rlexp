from abc import abstractmethod
from typing import Union, Callable, Dict, Tuple
from gymnasium import Env, Space
import numpy as np
import torch
from torch import nn
from tianshou.data.buffer.base import ReplayBuffer, Batch
from envs.gridworld.gridworld import Gridworld
from common import TabularAgent, Scheduler, train_loop


class SFQTabular(TabularAgent):
    def __init__(self, params:Dict, rb:ReplayBuffer, eps_scheduler:Scheduler, obs_to_inds: Callable, acts_to_inds: Callable, device) -> None:
        super().__init__(params=params)
        self.device = device

        self.rb: ReplayBuffer = rb
        self._obs_to_inds: Callable = obs_to_inds
        self._acts_to_inds: Callable = acts_to_inds
        
        # See Barreto et al. 2013
        self.d = (self.n_states**2 * self.n_acts).item()
        
        self.lr = params["step_size"]
        self.gamma = params["disc_fact"]

        self.warmup_steps = params["warmup_steps"]
        self.total_steps = 0
        
        self.epsilon_scheduler = eps_scheduler
        self.epsilon = None

        # \Psi \in R^{S, A, S^2*A} from Barreto et al. 2018. To be kept track of in the form of a table.
        self.psi = torch.zeros((self.n_states, self.n_acts, self.d), device=self.device)
        
        # w from Barreto et al. 2018. To be learned in a supervised fashion.
        self.w = torch.rand(size=(self.d, 1), requires_grad=True, device=self.device)
        # Initialise weights to some small random values
        self.w.data *= 0.001
        
        self.weight_keys = ['psi', 'w', 'q_table', 'v_table']

        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.SGD([self.w], lr=params["step_size"])
        self.done = False

        # Seeding random generators for reproducibility
        if params["seed"] is not None:
            self.rng = torch.Generator().manual_seed(params["seed"])
        else:
            self.rng = torch.Generator()
    
    @property
    def q_table(self) -> torch.Tensor:
        return torch.matmul(self.psi, self.w).detach()

    @property
    def v_table(self) -> torch.Tensor:
        return torch.mean(self.q_table, axis=1).reshape(env.grid_shape)

    @property
    def psi_ss(self) -> torch.Tensor:
        psi_s = torch.sum(self.psi, dim=1)
        psi_ss = torch.zeros((self.n_states, self.n_states), device=self.device)

        for s_from_ind in range(self.n_states):
            # psi_sparse \in R^d -- all transitions that ocurred after state s_from_ind. 
            # We want to sum up all the time we ended up in some state s_to.
            psi_sparse = psi_s[s_from_ind, :]
            for one_hot_index in range(self.d):
                _, _, s_to_ind = self._one_hot_index_to_dense_index(one_hot_index=one_hot_index)
                psi_ss[s_from_ind, s_to_ind] += psi_sparse[one_hot_index]
        return psi_ss
    
    def store_transition(self, obs:Union[torch.Tensor, np.ndarray], next_obs:Union[torch.Tensor, np.ndarray], action:Union[torch.Tensor, np.ndarray], reward:Union[torch.Tensor, np.ndarray], terminated: bool, truncated: bool) -> None:
        obs, next_obs, action, reward, terminated, truncated = super().store_transition(obs, next_obs, action, reward, terminated, truncated)
        transition = Batch(obs=obs, act=action, obs_next=next_obs, rew=reward, terminated=terminated, truncated=truncated)
        self.rb.add(transition)

    def _select_optimal_action(self, obs: torch.Tensor) -> int:
        q_sa = torch.empty(self.n_acts)
        obs_ind = self._obs_to_inds(obs)
        # Vectorized Q-value computation: shape (A, d) @ (d,)
        q_sa = torch.matmul(self.psi[obs_ind], self.w)  # Shape (A,)
        # Select the action with the highest Q-value
        act_ind = torch.argmax(q_sa).item()
        return act_ind

    def select_action(self, obs: torch.Tensor, update:bool=True) -> int:
        if update:
            self.epsilon = self.epsilon_scheduler.step()
            self.history["epsilon"].append(self.epsilon)
            self.total_steps += 1

        if torch.rand(1, generator=self.rng).item() < self.epsilon:
            act_ind = torch.randint(low=0, high=self.n_acts, size=(1,))
        else:
            act_ind = self._select_optimal_action(obs=obs)
        return act_ind
    
    def _update_psi(self, obs_ind:int, action_ind:int, next_obs_ind:int, next_action_ind:int) -> None:
        # Update (Barreto et al., 2018) Equation (4).
        one_hot_index = self._dense_index_to_one_hot_index(obs_ind=obs_ind, action_ind=action_ind, next_obs_ind=next_obs_ind)
        phi_t = self.one_hot_index_to_vector(one_hot_index=one_hot_index)
        expect_pi = self.psi[next_obs_ind, next_action_ind, :]
        self.psi[obs_ind, action_ind, :] = phi_t + self.gamma * expect_pi
        return phi_t
    
    def _update_w(self, phi_t:torch.Tensor, reward:float) -> None:
        # Update self.w in a supervised fashion
        r_pred = torch.matmul(phi_t, self.w)
        r_target = torch.unsqueeze(torch.tensor(reward, dtype=torch.float32, device=self.device), dim=0)
        loss = self.loss_fn(r_pred, r_target)
        
        self.history["w_loss"].append(loss.item())
        
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

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
            next_action = self.select_action(obs=transition.obs_next, update=False)
            next_action_ind = self._acts_to_inds(next_action)
            
            # Update \Psi(s,a) according to Barreto et al. 2018, Eq. (4)
            phi_t = self._update_psi(
                obs_ind=obs_ind, 
                action_ind=action_ind, 
                next_obs_ind=next_obs_ind,
                next_action_ind=next_action_ind
            )

            # Update w according to Barreto et al. 2018
            self._update_w(phi_t=phi_t, reward=transition.rew)

            i += 1
            if transition.terminated is True:
                i += 1
    
    def store_weights(self, path: str) -> Tuple:
        torch.save({'psi': self.psi.cpu(), 'w': self.w.cpu(), 'q_table': self.q_table.cpu(), 'v_table': self.v_table.cpu()}, path)
        return self.weight_keys
    
    def load_weights(self, path:str) -> None:
        weights = torch.load(path, weights_only=False)
        self.psi = weights[self.weight_keys[0]].to(self.device)
        self.w = weights[self.weight_keys[1]].to(self.device)
        return self.weight_keys


if __name__ == '__main__':
    import os
    from utils import setup_artefact_paths
    from plotting import plot_scalar
    import argparse

    parser = argparse.ArgumentParser(description="Script with a rerun flag")
    parser.add_argument("--rerun", action="store_true", default=True, help="Set to rerun the training process (default: True)")
    args = parser.parse_args()
    to_rerun = args.rerun

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path)
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    weights_path = os.path.join(store_path, f'{hparams["algo_type"]}_weights.pt')

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
    
    agent = SFQTabular(
        params=hparams,
        rb=rb, 
        eps_scheduler=scheduler,
        obs_to_inds=env.obs_to_ids,
        acts_to_inds=env.acts_to_ids,
        device=device
    )
    
    if to_rerun:
        _ = train_loop(
            env=env,
            agent=agent,
            hparams=hparams,
            random=False
        )
        weight_keys = agent.store_weights(path=weights_path)
    else:
        weight_keys = agent.load_weights(path=weights_path)

    # Plot V table for debugging
    env.plot_values(table=agent.v_table.cpu(), plot_name=f'{hparams["algo_type"]}_vtable')
    
    # Plot Psi(s) table for debugging. See Machado et al. 2021 Algorithm 1
    psi_ss = agent.psi_ss
    for state_id, state_obs in hparams["states"].items():
        # You must get the dense index here!
        obs_ind = env.obs_to_ids(torch.tensor(state_obs)).item()
        psi_s = psi_ss[obs_ind, :]
        psi_s = psi_s.reshape(env.grid_shape).cpu()
        env.plot_values(table=psi_s, plot_name=f'{hparams["algo_type"]}_state_{state_id}')
    
    loss_path = os.path.join(store_path, f'{hparams["algo_type"]}_loss')
    plot_scalar(
        scalars=agent.history["w_loss"], 
        save_path=loss_path
    )

    epsilon_path = os.path.join(store_path, f'{hparams["algo_type"]}_epsilon')
    plot_scalar(
        scalars=agent.history["epsilon"], 
        save_path=epsilon_path
    )