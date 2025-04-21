from copy import deepcopy
from typing import Dict, Tuple, Union
import numpy as np
import torch
from torch import nn
from common import Agent, Scheduler
from tianshou.data.buffer.base import ReplayBuffer, Batch
from transformers import BertTokenizer, BertModel

class SFBert(Agent):
    def __init__(self, params: Dict, phi_nn:nn.Module, psi_nn:nn.Module, n_acts:int, rb:ReplayBuffer, device):
        super().__init__(params=params)
        self.device = device

        self.rb: ReplayBuffer = rb

        # Set up BERT and ensure it isn't part of the backprop graph
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.w_nn = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(self.device)
        for param in self.w_nn.parameters():
            param.requires_grad = False
        self.w_nn.eval()

        # Psi(s,a) and Phi(s) share the same base. To obtain Psi(s,*) call phi_s = Phi(s) first
        # and then call Psi(phi_s). This gives Psi(s,a) for all a.
        self.phi_nn: nn.Module = phi_nn.to(self.device)
        self.psi_nn: nn.Module = psi_nn.to(self.device)
        self.psi_nn_t: nn.Module = deepcopy(psi_nn).to(self.device)
        for p in self.psi_nn_t.parameters():
            p.requires_grad = False
        self.psi_nn_t.eval()
        self.update_target_model()

        self.n_acts = n_acts

        self.instr_text = ""
        self.w = self.get_embedding(text=self.instr_text)

        self.lr = params["step_size"]
        self.gamma = params["disc_fact"]

        self.warmup_steps = params["warmup_steps"]
        self.total_steps = 0
        self.nn_t_update_steps = params["target_update_steps"]
        
        self.phi_optim = torch.optim.Adam(self.phi_nn.parameters(), lr=self.lr)
        self.psi_optim = torch.optim.Adam(self.psi_nn.parameters(), lr=self.lr)

        if params["seed"] is not None:
            self.rng = torch.Generator().manual_seed(params["seed"])
        else:
            self.rng = torch.Generator()
        self.weight_keys = ['phi_nn', 'psi_nn']

    def update_target_model(self):
        self.psi_nn_t.load_state_dict(self.psi_nn.state_dict())

    def store_transition(self, obs:Union[torch.Tensor, np.ndarray], next_obs:Union[torch.Tensor, np.ndarray], action:Union[torch.Tensor, np.ndarray], reward:Union[torch.Tensor, np.ndarray], terminated: bool, truncated: bool) -> None:
        obs, next_obs, action, reward, terminated, truncated = super().store_transition(obs, next_obs, action, reward, terminated, truncated)
        transition = Batch(
            obs=obs, 
            act=action, 
            obs_next=next_obs, 
            rew=reward, 
            terminated=terminated, 
            truncated=truncated, 
            info=Batch(w=deepcopy(self.w))
        )
        self.rb.add(transition)

    def get_embedding(self, text:str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.w_nn(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings.requires_grad = False
        return embeddings[0, 0, :].detach()
    
    def psi_update(self, batch: Batch) -> float:
        batch_size = len(batch)
        # Get the active instruction when the transition was played
        w = batch.info.w
        
        with torch.no_grad():
            phis = self.phi_nn(batch.obs)
        
        # Get the relevant Psi(s,a) vector for the stored transition. 
        # Psi outputs need to be converted to Q-values to get the optimal action to index Psi outputs
        # Psi(s,a) \in (batch_size, num_actions, psi_dim)
        
        psis = self.psi_nn(phis)
        # Q(s,a) \in (batch_size, num_actions)
        qs = torch.bmm(psis, w.unsqueeze(2)).squeeze(2)
        acts_selected = torch.argmax(qs, dim=1).squeeze(-1)
        psis_selected = psis[torch.arange(batch_size), acts_selected, :]

        # Get the relevant Psi(s',a') vector for the greedy action a' to be played in the transition next_state. 
        with torch.no_grad():
            # Do the same Psi -> Q conversion and Psi slicing as before, only for the next state in the transition this time
            phis_next = self.phi_nn(batch.obs_next)
            psis_next = self.psi_nn_t(phis_next)
            qs_next = torch.bmm(psis_next, w.unsqueeze(2)).squeeze(2)
            acts_greedy = torch.argmax(qs_next, dim=1).squeeze(-1)
            psis_next_greedy = psis_next[torch.arange(batch_size), acts_greedy, :]
            
            # Get Phi(s) (equivalent to the reward in the standard Bellman update)
            psis_target = phis + (1. - batch.terminated).unsqueeze(-1) * self.gamma * psis_next_greedy

        # We calculate the absolute difference between current and target values q values,
        # which is useful info for debugging.
        with torch.no_grad():
            td_error = torch.abs(psis_target - psis_selected)

        # We update the "live" network, self.current. First we zero out the optimizer gradients
        # and then we apply the update step using qs_selected and qs_target.
        self.psi_optim.zero_grad()
        loss = (torch.nn.functional.mse_loss(psis_selected, psis_target)).mean()
        loss.backward()
        self.psi_optim.step()
        return torch.mean(td_error).item()
    
    def phi_update(self, batch:Batch) -> float:
        r_target = batch.rew
        w = batch.info.w
        r_pred = torch.bmm(
            self.phi_nn(batch.obs).unsqueeze(1), 
            w.unsqueeze(2)
        ).squeeze()
    
        self.phi_optim.zero_grad()
        loss = (torch.nn.functional.mse_loss(r_pred, r_target)).mean()
        loss.backward()
        self.phi_optim.step()
        
        return loss.detach().item()
    
    def update_target_model(self):
        self.psi_nn_t.load_state_dict(self.psi_nn.state_dict())

    def update(self, batch_size:int=0) -> None:
        self.total_steps += 1

        if self.total_steps < self.warmup_steps:
            return
        
        batch, _ = self.rb.sample(batch_size=batch_size)
        batch.to_torch(device=self.device, dtype=torch.float32)
        
        td_error = self.psi_update(batch=batch)
        self.history["psi_td_error"].append(td_error)

        l2_error = self.phi_update(batch=batch)
        self.history["phi_l2_error"].append(l2_error)

        if self.total_steps % self.nn_t_update_steps == 0:
            self.update_target_model()    
        
    def store_weights(self):
        return super().store_weights()
    
    def load_weights(self):
        return super().load_weights()
    
    def _select_optimal_action(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            phi = self.phi_nn(obs)
            psi = self.psi_nn(phi)
            q = psi.T @ self.w
            opt_act = torch.argmax(q).squeeze(-1)
        return opt_act.detach().item()

    def select_action(self, obs: torch.Tensor, instr:str, epsilon:float=None) -> int:
        # If the instruction has changed, update it
        if instr != self.instr_text:
            self.instr_text = instr
            self.w = self.get_embedding(text=self.instr_text)

        if epsilon is not None and torch.rand(1, generator=self.rng).item() < epsilon:
            act_ind = torch.randint(low=0, high=self.n_acts, size=(1,)).item()
        else:
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32)
            obs = obs.to(self.device)
            act_ind = self._select_optimal_action(obs=obs)
        return act_ind

    def store_weights(self, path: str) -> Tuple:
        torch.save({'phi_nn': self.phi_nn.state_dict(), 'psi_nn': self.psi_nn.state_dict()}, path)
        return self.weight_keys
    
    def load_weights(self, path:str) -> None:
        weights = torch.load(path, weights_only=False)
        self.phi_nn.load_state_dict(weights[self.weight_keys[0]])
        self.phi_nn = self.phi_nn.to(self.device)
        self.psi_nn.load_state_dict(weights[self.weight_keys[1]])
        self.psi_nn = self.psi_nn.to(self.device)
        return self.weight_keys
    

if __name__ == '__main__':
    import os
    from utils import setup_artefact_paths
    from plotting import plot_scalar
    from envs.taxicab.mutlitask_taxicab import LanguageTaxicab
    from nets import FCMultiHead, FCTrunk
    import argparse
    from tqdm import tqdm

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

    env = LanguageTaxicab(
        hparams=hparams, 
        store_path=store_path
    )

    rb = ReplayBuffer(size=hparams['buffer_size'])
    
    phi_nn = FCTrunk(
        in_dim=env.observation_space.shape[0],
        h=[256, 768]
    )

    psi_nn = FCMultiHead(
        in_dim=768,
        num_heads=env.action_space.n,
        h=[768]
    )
    
    agent = SFBert(
        params=hparams,
        rb=rb, 
        n_acts=env.action_space.n,
        phi_nn=phi_nn,
        psi_nn=psi_nn,
        device=device
    )
    
    if to_rerun:
       returns = {}
       instr_eps = {}
       for _ in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset(options={"set_id": "train"})
        instruction = env.instruction
        if instruction not in instr_eps:
            instr_eps[instruction] = Scheduler(start=hparams['schedule_start'], end=hparams['schedule_end'], decay_func=lambda step: hparams['schedule_decay'] * step)
            returns[instruction] = []
        
        done = False
        total_reward = 0

        while not done:
            epsilon = instr_eps[instruction].step()
            action = agent.select_action(obs=obs, instr=env.instruction, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(
                obs=obs, 
                next_obs=next_obs,
                action=action, 
                reward=reward, 
                terminated=terminated, 
                truncated=truncated
            )
            agent.update(batch_size=hparams['batch_size'])
            
            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
        
        returns[instruction].append(total_reward)
        weight_keys = agent.store_weights(path=weights_path)
    else:
        weight_keys = agent.load_weights(path=weights_path)
    
    loss_path = os.path.join(store_path, f'{hparams["algo_type"]}_psi_td')
    plot_scalar(
        scalars=agent.history["psi_td_error"], 
        save_path=loss_path,
        label="Psi TD Bellman Error"
    )

    loss_path = os.path.join(store_path, f'{hparams["algo_type"]}_phi_l2')
    plot_scalar(
        scalars=agent.history["phi_l2_error"], 
        save_path=loss_path,
        label="Phi L2 Loss"
    )

    # epsilon_path = os.path.join(store_path, f'{hparams["algo_type"]}_epsilon')
    # plot_scalar(
    #     scalars=agent.history["epsilon"], 
    #     save_path=epsilon_path,
    #     label="Epsilon"
    # )
    i = 0
    for instr, rets in returns.items():
        loss_path = os.path.join(store_path, f'{hparams["algo_type"]}_return_{i}')
        plot_scalar(
            scalars=rets, 
            save_path=loss_path,
            label="Returns",
            title=instr
        )
        i+= 1

    
