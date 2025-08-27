import time
from copy import deepcopy
from itertools import chain
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from typing import Any, Dict, List, Optional, Tuple
from tianshou.data.buffer.base import Batch
from tianshou.data import ReplayBuffer
from tianshou.policy.base import TrainingStats
from tianshou.policy import BasePolicy
from torch.distributions import Categorical
from tianshou.utils.torch_utils import torch_train_mode
from algos.common import argmax_random_tiebreak, GroupedReplayBuffer


# TODO: Implement separate logging for separate psi losses. Implement smoothed psi loss as average of all losses!
@dataclass
class SFBaseTrainingStats(TrainingStats):
    psi_td_loss: float = 0.0
    psi_td_loss_max: float = 0.0
    phi_l2_loss: float = 0.0
    rec_loss: float = 0.0
    epsilon: float = 0.0
    terminal_freq: float = 0.0
    norm_phi: float = 0.0
    norm_psi: float = 0.0

# TODO: How does knowledge transfer occurr here exactly? When we encounter a new goal, shouldn't we copy the
# current parameters of the closest model we currently have?
class MultiparamModule(nn.Module):
    def __init__(self, base_module: nn.Module, num_modules: int, lr: float, tau:float, precomp_embeddings: Dict):
        super(MultiparamModule, self).__init__()
        self.precomp_embed = precomp_embeddings
        self.num_modules = num_modules
        self.tau = tau
        self._counters: List[int] = [0 for _ in range(num_modules)]

        self.nets = nn.ModuleList([deepcopy(base_module) for _ in range(num_modules)])
        self.t_nets = nn.ModuleList([deepcopy(base_module) for _ in range(num_modules)])

        for index, m in enumerate(self.t_nets):
            m.load_state_dict(self.nets[index].state_dict())
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        self._optims = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.nets]
        
        self._key_map = {}
        self._keys = set({})
    
    def _check_add_key(self, key:str) -> None:
        if key in self._keys:
            return
        
        key_index = len(self._keys)
        self._keys.add(key)
        self._key_map[key] = key_index
        
    def get_module(self, key:str, target=False) -> nn.Module:
        self._check_add_key(key=key)
        if target:
            return self.t_nets[self._key_map[key]]
        else:
            return self.nets[self._key_map[key]]
    
    def get_optim(self, key:str) -> torch.optim.Optimizer:
        self._check_add_key(key=key)
        return self._optims[self._key_map[key]]
    
    @property
    def counter(self):
        return self._counters
    
    @property
    def keys(self):
        return self._keys
    
    def get_counter(self, key:str) -> int:
        self._check_add_key(key=key)
        return self._counters[self._key_map[key]]
    
    def update_counter(self, key:str) -> int:
        self._check_add_key(key=key)
        self._counters[self._key_map[key]] += 1
    
    def soft_update_target(self, key:str) -> None:
        t_module = self.get_module(key=key, target=True)
        module = self.get_module(key=key, target=False)
        
        with torch.no_grad():
            for p_t, p in zip(t_module.parameters(), module.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def forward(self, x: torch.Tensor, state: Dict=None, **kwargs):
        target = False
        if state is None:
            state = {}
        
        if "target" in state and state["target"]:
            target = True
            
        if "key" in state:
            module = self.get_module(key=state["key"], target=target)
            result = module(x, **kwargs)
        else:
            result = torch.stack([self.get_module(key=k, target=target)(x, **kwargs).squeeze(0) for k in self._keys], dim=0)
        return result
    
    def get_extra_state(self) -> Dict[str, Any]:
        """
        Called by torch when building state_dict(); its return value is pickled
        into state_dict under the 'extra_state' key.
        """
        return {
            "keys": list(self._keys),
            "key_map": dict(self._key_map)
        }
    
    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """
        Called by torch.load_state_dict() after tensor parameters/buffers are loaded.
        Restore non-tensor Python state here.
        """
        keys = state.get("keys", [])
        key_map = state.get("key_map", {})

        if len(keys) == 0:
            raise ValueError("Trying to load 0 skills from the checkpoint. Minimum number of skills is 1.")
        
        self._keys = list(keys)
        self._key_map = dict(key_map)

    

class SFBase(BasePolicy):
    def __init__(
            self, 
            phi_nn:torch.nn.Module, 
            psi_nn:torch.nn.Module, 
            precomp_embeddings:Dict,
            rb:ReplayBuffer,
            num_skills,
            action_space, 
            psi_lr:float,
            phi_lr:float,
            psi_lambda:float,
            phi_lambda:float,
            psi_update_tau:float,
            phi_update_tau:float,
            phi_update_ratio:float, 
            l2_freq_scaling:bool, 
            gamma:float=0.99, 
            seed:float=1., 
            terminal_rew:float=20,
            dec_nn:Optional[torch.nn.Module]=None, 
            device:torch.device=torch.device("cpu")
        ):
        super().__init__(action_space=action_space)
        self.device = device

        self.phi_lr = phi_lr
        self.psi_lr = psi_lr
        self.gamma = gamma
        self.rb = rb

        # Psi(s,a) and Phi(s) share the same base. To obtain Psi(s,*) call phi_s = Phi(s) first
        # and then call Psi(phi_s). This gives Psi(s,a) for all a.
        self.phi_nn: nn.Module = phi_nn
        self.phi_nn_t = deepcopy(self.phi_nn).to(self.device).eval()
        for p in self.phi_nn_t.parameters(): 
            p.requires_grad = False
        self.phi_optim = torch.optim.Adam(self.phi_nn.parameters(), lr=self.phi_lr)
        
        self.phi_update_tau = phi_update_tau
        self.phi_update_freq = phi_update_ratio

        self.precomp_embed = precomp_embeddings
        for key in self.precomp_embed.keys():
            self.precomp_embed[key] = self.precomp_embed[key] / (self.precomp_embed[key].norm(p=2) + 1e-8)
            self.precomp_embed[key] = self.precomp_embed[key].to(self.device)
        
        self.psi_nn: MultiparamModule = MultiparamModule(
            base_module=psi_nn, 
            num_modules=num_skills, 
            lr=self.psi_lr,
            tau=psi_update_tau,
            precomp_embeddings=self.precomp_embed
        )
        
        self.dec_nn = dec_nn
        if self.dec_nn is not None:
            self.use_reconstruction_loss = True
            self.dec_optim = torch.optim.Adam(self.dec_nn.parameters(), lr=self.phi_lr)
        else:
            self.use_reconstruction_loss = False
            self.dec_optim = None

        self.update_count = 0  # Counter for training iterations
        self.r_counter = Counter()
        self.l2_freq_scaling = l2_freq_scaling

        self.phi_lambda = phi_lambda
        self.psi_lambda = psi_lambda
        
        self.phi_l2_loss = 0
        self.psi_td_loss = 0
        self.rec_loss = 0

        # For logging and debugging purposes
        self.terminal_rew = terminal_rew
        self.terminal_freq = 0

        # To be set by trainer
        self.eps = None
        self.max_action_num = self.action_space.n
        
        self.rng = torch.Generator().manual_seed(seed)

    @property
    def instructions(self):
        return self.psi_nn.keys
    
    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps
    
    def instr_to_embedding(self, instrs) -> torch.Tensor:
        w = torch.stack(
                [self.precomp_embed[instr] for instr in instrs]
            ).to(self.device)
        return w
    
    def _soft_update_phi_target(self):
        with torch.no_grad():
            for p_t, p in zip(self.phi_nn_t.parameters(), self.phi_nn.parameters()):
                p_t.data.mul_(1 - self.phi_update_tau).add_(self.phi_update_tau * p.data)

    def forward(self, batch, state=None, **kwargs):
        """Does a forward pass with the agent on a batch of observations. 
        Cannot assume all observations share the same instruction. 
        In most cases receives batch of size 1.

        Args:
            batch (tiansou.data.Batch): Batch of observations.
            state (_type_, optional): Additional info. Defaults to None.

        Returns:
            tianshou.data.Batch: Batch of actions.
        """
    
        with torch.no_grad():
            numerical_features = batch.obs.features
            # Retrieve embeddings for instructions
            w = self.instr_to_embedding(instrs=batch.obs.instr)

            parse_batched = False
            if len(batch) == 1 or (w == w[0]).all().item():
                parse_batched = True
            
            # shape: (batch_dim, num_actions, embedding_dim)
            phi_sa = self.phi_nn(numerical_features)
            _, num_actions, _ = phi_sa.shape

            # If a single-element batch or all elements are the same, 
            # we can save on some costly operations by processing it at once
            if parse_batched:
                phi_a = torch.chunk(phi_sa, chunks=num_actions, dim=1)
                st = {"key": str(batch.obs.instr[0])}
                psi_a = [self.psi_nn.forward(chunk.squeeze(1), st).unsqueeze(1) for chunk in phi_a]
            else:
                # If elements belong to different instructions, we need to use different parameterisations for psi network. 
                # This means we need to parse each element separately, hence the loop below.
                psi_a = []
                for index in range(len(batch)):
                    transition = batch[index]
                    key = str(batch.obs.instr[index])
                    st = {"key": key}
                    # transition shape: (1, num_actions, embedding_dim)
                    # shape: [(1, 1, embedding_dim), ...]
                    phi_a_single = torch.chunk(transition, chunks=num_actions, dim=1)
                    psi_a_single = [self.psi_nn.forward(chunk.squeeze(0).squeeze(0), st).unsqueeze(0).unsqueeze(0)
                                     for chunk in phi_a_single]
                    psi_a_single = torch.cat(psi_a_single, dim=0)
                    psi_a.append(psi_a_single)
            
            # shape: (batch_dim, num_actions, embedding_dim)
            psi = torch.cat(psi_a, dim=1)

            # shape: (batch_dim, num_actions)
            q_logits = torch.bmm(psi, w.unsqueeze(2)).squeeze(2)
            dist = Categorical(logits=q_logits)
            act = argmax_random_tiebreak(q_logits)
            act = dist.sample()
        return Batch(act=act, state=state, dist=dist)
    
    def forward_gpi(self, batch, state, **kwargs):
        num_skills = len(self.psi_nn.keys)
        numerical_features = batch.obs.features
        # Retrieve embeddings for instructions
        w = self.instr_to_embedding(instrs=batch.obs.instr)
        w = w.repeat(num_skills, 1)
        
        # shape: (batch_dim, num_actions, embedding_dim)
        phi_sa = self.phi_nn(numerical_features)
        _, num_actions, _ = phi_sa.shape
        
        phi_a = torch.chunk(phi_sa, chunks=num_actions, dim=1)
        # Does inference with all task policies: num_a x num_skills list
        psi_all = torch.stack([self.psi_nn.forward(chunk.squeeze(1), {}) for chunk in phi_a], dim=0)
        # Now transpose to (num_skills, num_a, feat_dim)
        psi_all = psi_all.transpose(0, 1)
        
        # (num_skills, num_a)
        q_logits = torch.bmm(psi_all, w.unsqueeze(2)).squeeze(2)
        # Best action per skill and its value
        skill_best_vals, skill_best_actions = q_logits.max(dim=1)
        # Choose the skill whose best value is globally maximal
        best_skill = skill_best_vals.argmax()
        # Global best action across all skills
        single_a = skill_best_actions[best_skill] 
        return Batch(act=single_a, state=state)
    
    def psi_update(self, batch: Batch) -> Tuple[float, Tuple]:
        batch_size = len(batch)
        if not isinstance(batch.terminated, torch.Tensor):
            terminated = torch.tensor(batch.terminated, dtype=torch.int).to(self.device)
        else:
            terminated = batch.terminated
        
        if not isinstance(batch.truncated, torch.Tensor):
            truncated = torch.tensor(batch.truncated, dtype=torch.int).to(self.device)
        else:
            truncated = batch.truncated
        done = (terminated | truncated).to(torch.float32) 

        if not isinstance(batch.act, torch.Tensor):
            acts_selected = torch.tensor(batch.act, dtype=torch.int).to(self.device)
        else:
            acts_selected = batch.act
        
        with torch.no_grad():
            # Get the active instruction when the transition was played
            w = self.instr_to_embedding(instrs=batch.obs.instr)
            key = str(batch.obs.instr[0])
            st = {"key": key}
            
            # Get phi(s,a) for each action a \in A for a given state s
            phi_sa = self.phi_nn_t(batch.obs.features)
            phis_selected = phi_sa[torch.arange(batch_size), acts_selected, :]
        _, num_actions, _ = phi_sa.shape
        
        # Run each phi_sa through the psi network individually to obtain successor features
        phi_a = torch.chunk(phi_sa, chunks=num_actions, dim=1)
        psi_a = [self.psi_nn.forward(chunk.squeeze(1), st).unsqueeze(1) for chunk in phi_a]
        # shape: (batch_dim, num_actions, embedding_dim)
        psis = torch.cat(psi_a, dim=1)
        
        # Psi outputs need to be converted to Q-values to get the optimal action to index Psi outputs
        # shape (batch_size, embedding_dim)
        psis_selected = psis[torch.arange(batch_size), acts_selected, :]

        # Get the relevant Psi(s',a') vector for the greedy action a' to be played in the transition next_state. 
        with torch.no_grad():
            st_t = {"key": key, "target": True}
            phi_sa_next = self.phi_nn_t(batch.obs_next.features)
            phi_next_a = torch.chunk(phi_sa_next, chunks=num_actions, dim=1)
            psis_next = [self.psi_nn.forward(chunk.squeeze(1), st_t).unsqueeze(1) for chunk in phi_next_a]
            psis_next = torch.cat(psis_next, dim=1)
            
            qs_next = torch.bmm(psis_next, w.unsqueeze(2)).squeeze(2)
            acts_greedy = argmax_random_tiebreak(qs=qs_next)
            psis_next_greedy = psis_next[torch.arange(batch_size), acts_greedy, :]
            
            # Get Phi(s) (equivalent to the reward in the standard Bellman update)
            psis_target = phis_selected + (1. - done).unsqueeze(-1) * self.gamma * psis_next_greedy
        
        # For debugging purposes
        phi_norm = phis_selected.norm(dim=-1).mean().item()
        psi_tgt_norm = psis_target.norm(dim=-1).mean().item()

        # Huber loss
        self.psi_nn.get_optim(key=key).zero_grad()
        huber_per_dim = nn.functional.smooth_l1_loss(psis_selected, psis_target, reduction='none')
        td_loss_per_sample = huber_per_dim.sum(dim=1)
        
        if hasattr(batch, "weight"):
            w_is = torch.as_tensor(batch.weight, device=self.device, dtype=td_loss_per_sample.dtype)
            w_is = w_is.clamp(max=5.0) # cap PER weights
            w_is = w_is / (w_is.mean().clamp(min=1e-8))
            loss = (w_is * td_loss_per_sample).mean()
        else:
            loss = td_loss_per_sample.mean()
        
        # Output norm regularisation on psi
        psi_pen = self.psi_lambda * psis_selected.pow(2).sum(-1).mean()
        total_loss = loss + psi_pen
        
        total_loss.backward()
        clip_grad_norm_(self.psi_nn.get_module(key=key).parameters(), max_norm=10)
        self.psi_nn.get_optim(key=key).step()
        self.psi_nn.soft_update_target(key=key)
        return loss.detach().cpu().numpy(), phi_norm, psi_tgt_norm, key
    
    def phi_update(self, batch:Batch) -> float:
        batch_size = len(batch)
        if not isinstance(batch.rew, torch.Tensor):
            r_target = torch.tensor(batch.rew, dtype=torch.float32).to(self.device)
        else:
            r_target = batch.rew

        if not isinstance(batch.act, torch.Tensor):
            acts_selected = torch.tensor(batch.act, dtype=torch.int).to(self.device)
        else:
            acts_selected = batch.act

        if not isinstance(batch.obs_next.features, torch.Tensor):
            obs_next = torch.tensor(batch.obs_next.features, dtype=torch.float32).to(self.device)
        else:
            obs_next = batch.obs_next.features
        
        values, counts = torch.unique(r_target.view(-1), return_counts=True)
        target_list = r_target.tolist()
        
        if self.l2_freq_scaling:
            r_counter = Counter(dict(zip(values.cpu().tolist(), counts.cpu().tolist())))
            self.r_counter += r_counter
            weights = torch.tensor([1.0 / self.r_counter[val] for val in target_list]).to(self.device)
            weights = weights / weights.sum()
        else:
            weights = torch.tensor([1.0 / len(target_list) for _ in target_list]).to(self.device)

        if self.terminal_rew in self.r_counter:
            self.terminal_freq = self.r_counter[self.terminal_rew] / sum(self.r_counter.values())

        w = self.instr_to_embedding(instrs=batch.obs.instr)

        self.phi_optim.zero_grad()
        if self.use_reconstruction_loss:
            self.dec_optim.zero_grad()

        phi_sa = self.phi_nn(batch.obs.features)
        phis_selected = phi_sa[torch.arange(batch_size), acts_selected, :]
        r_pred = torch.bmm(phis_selected.unsqueeze(1), w.unsqueeze(2)).squeeze()

        reward_errors = (r_pred - r_target).pow(2)
        reward_loss = (weights * reward_errors).sum()

        if self.use_reconstruction_loss:
            # Decoder network in this context is the forward dynamics prediction
            # s_hat = dec(phi(s,a))
            s_next_hats = self.dec_nn(phis_selected)
            rec_loss = (s_next_hats - obs_next).pow(2).mean()
            total_loss = reward_loss + rec_loss
        else:
            rec_loss = torch.tensor(0., device=self.device)
            total_loss = reward_loss
        
        # Output norm regularisation on phi
        phi_pen = self.phi_lambda * (phis_selected.pow(2).sum(dim=-1).mean())
        total_loss += phi_pen

        total_loss.backward()

        # clip
        clip_grad_norm_(self.phi_nn.parameters(), 10)
        if self.use_reconstruction_loss:
            clip_grad_norm_(self.dec_nn.parameters(), 10)

        # step
        self.phi_optim.step()
        if self.use_reconstruction_loss:
            self.dec_optim.step()
        self._soft_update_phi_target()
        
        return reward_loss.detach().item(), rec_loss.detach().item()
    
    def update(self, sample_size, buffer: GroupedReplayBuffer, **kwargs):
        # Increment the iteration counter
        self.update_count += 1

        start_time = time.time()

        if self.update_count % int(1/self.phi_update_freq) == 0:
            phi_batch, phi_indices = buffer.sample(sample_size)
            phi_batch = self.process_fn(phi_batch, buffer, phi_indices)
        else:
            phi_batch = None

        # All samples in a batch need to have the same instruction
        # so they could be processed as a batch.
        # TODO: Current code assumes a 1:1 mapping between instructions and corresponding skills. However,
        # if we ever want to map multiple instructions to the same skill (rephrased instructions, for example)
        # we should index replay buffer by instruction embedding not instruction id.
        psi_batches = []
        psi_indicess = []
        for key in self.psi_nn.keys:
            try:
                psi_batch, psi_indices = buffer.sample_group(
                    group_id=key, 
                    batch_size=sample_size
                )
                psi_batch = self.process_fn(psi_batch, buffer, psi_indices)
                psi_batches.append(psi_batch)
                psi_indicess.append(psi_indices)
            except ValueError:
                pass

        self.updating = True
        
        with torch_train_mode(self):
            training_stat = self.learn(phi_batch, psi_batches, **kwargs)
        
        if phi_batch is not None:
            self.post_process_fn(phi_batch, buffer, phi_indices)

        for batch, indices in zip(psi_batches, psi_indicess):
            self.post_process_fn(batch, buffer, indices)
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        
        training_stat.train_time = time.time() - start_time
        
        return training_stat

    def learn(self, phi_batch, psi_batches, **kwargs):
        
        if phi_batch is not None:
            self.phi_l2_loss, self.rec_loss = self.phi_update(batch=phi_batch)
        
        td_losses = []
        phi_norms = []
        psi_norms = []
        td_loss_map = {}

        for batch in psi_batches:
            td_error, phi_norm, psi_norm, instr_id = self.psi_update(batch=batch)
            td_losses.append(td_error.mean())
            phi_norms.append(phi_norm)
            psi_norms.append(psi_norm)
            td_loss_map[instr_id] = td_error
            self.psi_nn.update_counter(key=instr_id)
        
        self.psi_td_loss = np.mean(td_losses)
        self.phi_norm = np.mean(phi_norms)
        self.psi_norm = np.mean(psi_norms)
        
        stats = SFBaseTrainingStats()
        stats.epsilon = self.eps
        stats.phi_l2_loss = self.phi_l2_loss
        stats.psi_td_loss = self.psi_td_loss
        stats.terminal_freq = self.terminal_freq
        stats.rec_loss = self.rec_loss
        # Debugging
        stats.norm_phi = self.phi_norm
        stats.norm_psi = self.psi_norm
        
        stats.psi_td_loss_max = np.amax(td_losses)
        for k, v in td_loss_map.items():
            setattr(stats, f"psi_td_per_instr/{k}", float(v))

        return stats

    def process_fn(self, batch, buffer, indices):
        batch.indices = indices
        return batch

    def exploration_noise(self, act, batch):
        # Ensure epsilon is provided in the batch
        batch_size = len(act)
        # Generate a random mask using torch's RNG
        rand_mask = torch.rand(batch_size, generator=self.rng) < self.eps
        rand_mask = rand_mask.cpu().numpy()
        # Generate random actions using torch's RNG
        rand_act = torch.randint(0, self.max_action_num, (batch_size,), generator=self.rng)
        rand_act = rand_act.cpu().numpy()
        act[rand_mask] = rand_act[rand_mask]
        return act
    

class SFMix(SFBase):
    """
    SFBase with learnable scalar-mixed instruction embeddings.

    IMPORTANT: Pass `precomp_embeddings` as:
        Dict[str, List[Tensor(d,)]]
    where each value is the list of per-layer embeddings for that instruction.
    """
    def __init__(
        self,
        phi_nn: nn.Module,
        psi_nn: nn.Module,
        mix_nn: nn.Module,
        layers_to_mix: Tuple[int],
        precomp_embeddings: Dict[str, List[torch.Tensor]],
        rb,
        num_skills,
        action_space,
        psi_lr: float,
        phi_lr: float,
        psi_lambda: float,
        phi_lambda: float,
        psi_update_tau: float,
        phi_update_tau: float,
        phi_update_ratio: float,
        l2_freq_scaling: bool,
        gamma:float=0.99,
        seed:float=1.0,
        terminal_rew:float=20,
        renorm_w:bool=False,
        dec_nn:Optional[nn.Module]=None,
        device:torch.device=torch.device("cpu"),
    ):
        # Call SFBase as-is. It will store precomp_embeddings on self.precomp_embed,
        # but we will not use SFBase's value-normalization; we override instr_to_embedding.
        super().__init__(
            phi_nn=phi_nn,
            psi_nn=psi_nn,
            precomp_embeddings=precomp_embeddings,
            rb=rb,
            num_skills=num_skills,
            action_space=action_space,
            psi_lr=psi_lr,
            phi_lr=phi_lr,
            psi_lambda=psi_lambda,
            phi_lambda=phi_lambda,
            psi_update_tau=psi_update_tau,
            phi_update_tau=phi_update_tau,
            phi_update_ratio=phi_update_ratio,
            l2_freq_scaling=l2_freq_scaling,
            gamma=gamma,
            seed=seed,
            terminal_rew=terminal_rew,
            dec_nn=dec_nn,
            device=device,
        )
        self.mix_nn: ScalarMix = mix_nn
        self._layers_to_mix = tuple(layers_to_mix)
        self._renorm_w = renorm_w
        
        self._precomp_embed_per_layer: Dict[str, List[torch.Tensor]] = {}
        
        mat: torch.Tensor
        for instr, mat in self.precomp_embed.items():
            mat = mat.to(self.device)
            selected: List[torch.Tensor] = []
            for li in self._layers_to_mix:
                v = mat[li]  # (d,)
                v = v / (v.norm(p=2) + 1e-8)
                selected.append(v)
            self._precomp_embed_per_layer[str(instr)] = selected
        
        del self.phi_optim
        self.phi_optim = torch.optim.Adam(
            chain(self.phi_nn.parameters(), self.mix_nn.parameters()),
            lr=self.phi_lr,
        )

    def instr_to_embedding(self, instrs) -> torch.Tensor:
        # Build list of layer-batch tensors: [E^l1, E^l2, ...], each of shape (B, d)
        layer_batches: List[torch.Tensor] = []

        # Pre-collect layer vectors for all instructions to keep order consistent
        # For layer k (in 0..K-1), stack over batch
        for k in range(len(self._layers_to_mix)):
            layer_k_batch = torch.stack(
                [self._precomp_embed_per_layer[str(instr)][k] for instr in instrs], dim=0  # (B, d)
            )
            layer_batches.append(layer_k_batch)

        # Differentiable mixture via provided mix_nn (assumed ScalarMix-like)
        w = self.mix_nn(layer_batches)  # (B, d)

        if self._renorm_w:
            w = w / (w.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return w


if __name__ == '__main__':
    import os
    from utils import setup_artefact_paths, setup_experiment
    from yaml_utils import load_yaml
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    from tianshou.data import Collector, ReplayBuffer
    from algos.common import EpsilonDecayHook, SaveHook, SFTrainer, BetaAnnealHook, CompositeHook
    from envs.shapes.multitask_shapes import MultitaskShapes, ShapesPositionFactory
    from algos.nets import precompute_bert_embeddings, extract_bert_layer_embeddings, FCTrunk, FCTree, ScalarMix
    
    torch.autograd.set_detect_anomaly(True)

    script_path = os.path.abspath(__file__)
    store_path, config_path = setup_artefact_paths(script_path=script_path)
    _, store_path, precomp_path = setup_experiment(store_path=store_path, config_path=config_path)
    with open(config_path, 'r') as file:
        hparams = load_yaml(file)

    exp_hparams = hparams["experiment"]
    env_hparams = hparams["environment"]
    seed = hparams["general"]["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    writer = SummaryWriter(store_path)
    logger = TensorboardLogger(writer)

    env_factory = ShapesPositionFactory(
        hparams=env_hparams, 
        store_path=store_path
    )
    train_env: MultitaskShapes = env_factory.get_env(set_id='TRAIN')
    # TODO: Measure success on holdout later, for now success is measured on TRAIN (we are not measuring generalisation)
    test_env: MultitaskShapes = env_factory.get_env(set_id='TRAIN', purpose='EVAL')
    
    all_instructions = env_factory.get_all_instructions()

    embedding_path = os.path.join(precomp_path, 'bert_embeddings.pt')
    if os.path.isfile(embedding_path):
        precomp_embeddings = torch.load(embedding_path, map_location=device)
    else:
        precomp_embeddings = precompute_bert_embeddings(all_instructions, device=device)
        torch.save(precomp_embeddings, embedding_path)
    
    if "bert_layer_index" in exp_hparams:
        bert_layer_ind = exp_hparams["bert_layer_index"]
    else:
        bert_layer_ind = -1
    
    if not isinstance(bert_layer_ind, list):
        layer_embeddings = extract_bert_layer_embeddings(embedding_dict=precomp_embeddings, layer_ind=bert_layer_ind)
    else:
        layer_embeddings = precomp_embeddings
    
    in_dim = train_env.observation_space["features"].shape[0] + layer_embeddings[next(iter(layer_embeddings))].shape[0]
    
    rb = GroupedReplayBuffer(size=exp_hparams['buffer_size'])

    phi_nn = FCTree(
        in_dim=train_env.observation_space["features"].shape,
        num_heads=train_env.action_space.n,
        h_trunk=exp_hparams["phi_trunk_dim"],
        h_head=exp_hparams["phi_head_dim"],
        device=device
    )

    psi_nn = FCTrunk(
        in_dim=exp_hparams["phi_head_dim"][-1] if isinstance(exp_hparams["phi_head_dim"], list) else exp_hparams["phi_head_dim"],
        h=exp_hparams["psi_nn_dim"] if isinstance(exp_hparams["psi_nn_dim"], list) else [exp_hparams["psi_nn_dim"]],
        device=device
    )

    if exp_hparams["use_reconstruction_loss"]:
        # Reverse layers of the phi_nn (which acts as an encoder)
        hidden_dim = deepcopy(exp_hparams["phi_nn_dim"])
        hidden_dim.reverse()
        in_dim = hidden_dim.pop(0)
        hidden_dim.append(train_env.observation_space["features"].shape[0])
        
        dec_nn = FCTrunk(
            in_dim=in_dim,
            h=hidden_dim,
            device=device
        )
    else:
        dec_nn = None
    
    if not isinstance(bert_layer_ind, list):
        agent = SFBase(
            phi_nn=phi_nn, 
            psi_nn=psi_nn,
            dec_nn=None,
            rb=rb,
            num_skills=len(all_instructions),
            action_space=train_env.action_space,
            precomp_embeddings=layer_embeddings,
            l2_freq_scaling=exp_hparams["l2_freq_scaling"],
            phi_lr=exp_hparams["phi_lr"],
            psi_lr=exp_hparams["psi_lr"],
            psi_lambda=exp_hparams["psi_lambda"],
            phi_lambda=exp_hparams["phi_lambda"],
            psi_update_tau=exp_hparams["psi_update_tau"],
            phi_update_tau=exp_hparams["phi_update_tau"],
            phi_update_ratio=exp_hparams["phi_update_ratio"],
            gamma=env_hparams["disc_fact"],
            seed=seed,
            device=device
        )
    else:
        mix_nn = ScalarMix(len(bert_layer_ind), trainable=True).to(device)
        agent = SFMix(
            phi_nn=phi_nn, 
            psi_nn=psi_nn,
            mix_nn=mix_nn,
            layers_to_mix=bert_layer_ind,
            dec_nn=None,
            rb=rb,
            num_skills=len(all_instructions),
            action_space=train_env.action_space,
            precomp_embeddings=layer_embeddings,
            l2_freq_scaling=exp_hparams["l2_freq_scaling"],
            phi_lr=exp_hparams["phi_lr"],
            psi_lr=exp_hparams["psi_lr"],
            psi_lambda=exp_hparams["psi_lambda"],
            phi_lambda=exp_hparams["phi_lambda"],
            psi_update_tau=exp_hparams["psi_update_tau"],
            phi_update_tau=exp_hparams["phi_update_tau"],
            phi_update_ratio=exp_hparams["phi_update_ratio"],
            gamma=env_hparams["disc_fact"],
            seed=seed,
            device=device
        )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
    train_collector.reset()
    train_collector.collect(n_step=exp_hparams["warmup_steps"], random=True)

    test_collector = Collector(agent, test_env, exploration_noise=True)
    
    n_epochs = exp_hparams["n_epochs"]
    n_steps = exp_hparams["epoch_steps"]

    hooks = CompositeHook(agent=agent, logger=logger, hooks=[])
    epoch_hook = EpsilonDecayHook(hparams=exp_hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)
    hooks.add_hook(epoch_hook)

    if exp_hparams["prioritised_replay"]:
        beta_anneal_hook = BetaAnnealHook(
            agent=agent, 
            buffer=rb, 
            beta_start=exp_hparams["priority_beta_start"],
            beta_end=exp_hparams["priority_beta_end"], 
            frac=exp_hparams["priority_beta_frac"],
            max_steps=n_epochs*n_steps,
            logger=logger
        )
        hooks.add_hook(beta_anneal_hook)
    save_hook_factory = SaveHook(save_path=f'{store_path}/best_model.pth')
    
    result = SFTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=exp_hparams["step_per_collect"],
        update_per_step=exp_hparams["update_per_step"], episode_per_test=exp_hparams["episode_per_test"], batch_size=exp_hparams["batch_size"],
        train_fn=hooks.hook,
        test_fn=lambda epoch, global_step: agent.set_eps(exp_hparams["test_epsilon"]),
        save_best_fn=save_hook_factory.hook,
        logger=logger
    ).run()
    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')