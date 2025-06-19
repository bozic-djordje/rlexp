from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from typing import Dict
from tianshou.data.buffer.base import Batch
from tianshou.policy.base import TrainingStats
from tianshou.policy import BasePolicy
from torch.distributions import Categorical
from algos.common import argmax_random_tiebreak

@dataclass
class SFBertTrainingStats(TrainingStats):
    psi_td_loss: float = 0.0
    phi_l2_loss: float = 0.0
    epsilon: float = 0.0
    terminal_freq: float = 0.0


class SFBert(BasePolicy):
    def __init__(self, phi_nn: torch.nn.Module, psi_nn: torch.nn.Module, lr:float, target_update_freq:int, action_space, precomp_embeddings:Dict, cycle_update_freq:int=500, gamma:float=0.99, seed:float=1., terminal_rew:float=20, global_weighing:bool=True, device:torch.device=torch.device("cpu")):
        super().__init__(action_space=action_space)
        self.device = device

        # Psi(s,a) and Phi(s) share the same base. To obtain Psi(s,*) call phi_s = Phi(s) first
        # and then call Psi(phi_s). This gives Psi(s,a) for all a.
        self.phi_nn: nn.Module = phi_nn
        self.psi_nn: nn.Module = psi_nn
        self.psi_nn_t: nn.Module = deepcopy(psi_nn)
        for p in self.psi_nn_t.parameters():
            p.requires_grad = False
        self.psi_nn_t.eval()
        self.sync_weight()
        self.update_phi = True

        self.precomp_embed = precomp_embeddings
        for key in self.precomp_embed:
            self.precomp_embed[key] = self.precomp_embed[key].to(self.device)

        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.cycle_update_freq = cycle_update_freq
        self.update_count = 0  # Counter for training iterations
        self.r_counter = Counter()
        self.global_weighing = global_weighing
        
        self.phi_optim = torch.optim.Adam(self.phi_nn.parameters(), lr=self.lr)
        self.psi_optim = torch.optim.Adam(self.psi_nn.parameters(), lr=self.lr)

        self.phi_l2_loss = 0
        self.psi_td_loss = 0

        # For logging and debugging purposes
        self.terminal_rew = terminal_rew
        self.terminal_freq = 0

        # To be set by trainer
        self.eps = None
        self.max_action_num = self.action_space.n
        
        self.rng = torch.Generator().manual_seed(seed)
    
    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.psi_nn_t.load_state_dict(self.psi_nn.state_dict())
    
    def instr_to_embedding(self, instrs) -> torch.Tensor:
        w = torch.stack(
                [self.precomp_embed[instr] for instr in instrs]
            ).to(self.device)
        return w

    def forward(self, batch, state=None, **kwargs):
        with torch.no_grad():
            numerical_features = batch.obs.features
            # Retrieve embeddings for instructions
            w = self.instr_to_embedding(instrs=batch.obs.instr)
            
            # shape: (batch_dim, phi_dim)
            phi = self.phi_nn(numerical_features)
            # shape: (batch_dim, num_actions, embedding_dim)
            psi = self.psi_nn(phi)
            # shape: (batch_dim, num_actions)
            q_logits = torch.bmm(psi, w.unsqueeze(2)).squeeze(2)
            
            dist = Categorical(logits=q_logits)
            act = dist.sample()
        return Batch(act=act, state=state, dist=dist)
    
    def psi_update(self, batch: Batch) -> float:
        batch_size = len(batch)
        if not isinstance(batch.terminated, torch.Tensor):
            terminated = torch.tensor(batch.terminated, dtype=torch.int).to(self.device)
        else:
            terminated = batch.terminated

        if not isinstance(batch.act, torch.Tensor):
            acts_selected = torch.tensor(batch.act, dtype=torch.int).to(self.device)
        else:
            acts_selected = batch.act

        # Get the active instruction when the transition was played
        w = self.instr_to_embedding(instrs=batch.obs.instr)
        
        with torch.no_grad():
            phis = self.phi_nn(batch.obs.features)
        
        # Get the relevant Psi(s,a) vector for the stored transition. 
        # Psi outputs need to be converted to Q-values to get the optimal action to index Psi outputs
        # Psi(s,a) \in (batch_size, num_actions, psi_dim)
        
        psis = self.psi_nn(phis)
        # Q(s,a) \in (batch_size, num_actions)
        psis_selected = psis[torch.arange(batch_size), acts_selected, :]

        # Get the relevant Psi(s',a') vector for the greedy action a' to be played in the transition next_state. 
        with torch.no_grad():
            # Do the same Psi -> Q conversion and Psi slicing as before, only for the next state in the transition this time
            phis_next = self.phi_nn(batch.obs_next.features)
            psis_next = self.psi_nn_t(phis_next)
            qs_next = torch.bmm(psis_next, w.unsqueeze(2)).squeeze(2)
            acts_greedy = argmax_random_tiebreak(qs=qs_next)
            psis_next_greedy = psis_next[torch.arange(batch_size), acts_greedy, :]
            
            # Get Phi(s) (equivalent to the reward in the standard Bellman update)
            psis_target = phis + (1. - terminated).unsqueeze(-1) * self.gamma * psis_next_greedy

        # We calculate the absolute difference between current and target values q values,
        # which is useful info for debugging.
        with torch.no_grad():
            td_error = torch.abs(psis_target - psis_selected)

        # We update the "live" network, self.current. First we zero out the optimizer gradients
        # and then we apply the update step using qs_selected and qs_target.
        self.psi_optim.zero_grad()
        loss = (torch.nn.functional.mse_loss(psis_selected, psis_target)).mean()
        loss.backward()
        clip_grad_norm_(self.psi_nn.parameters(), max_norm=10)
        self.psi_optim.step()
        return torch.mean(td_error).item()
    
    def phi_update(self, batch:Batch) -> float:
        if not isinstance(batch.rew, torch.Tensor):
            r_target = torch.tensor(batch.rew, dtype=torch.float32).to(self.device)
        else:
            r_target = batch.rew
        
        values, counts = torch.unique(r_target.view(-1), return_counts=True)
        target_list = r_target.tolist()
        
        if self.global_weighing:
            r_counter = Counter(dict(zip(values.cpu().tolist(), counts.cpu().tolist())))
            self.r_counter += r_counter
            weights = torch.tensor([1.0 / self.r_counter[val] for val in target_list]).to(self.device)
            weights = weights / weights.sum()
        else:
            inv_counts = 1.0 / counts.cpu().float()
            w_map = dict(zip(values.cpu().tolist(), inv_counts.tolist()))
            weights = torch.tensor([w_map[val] for val in target_list]).to(self.device)

        if self.terminal_rew in self.r_counter:
            self.terminal_freq = self.r_counter[self.terminal_rew] / sum(self.r_counter.values())

        w = self.instr_to_embedding(instrs=batch.obs.instr)
        
        r_pred = torch.bmm(
            self.phi_nn(batch.obs.features).unsqueeze(1), 
            w.unsqueeze(2)
        ).squeeze()

        self.phi_optim.zero_grad()
        squared_errors = (r_pred - r_target) ** 2
        weighted_loss = (weights * squared_errors).sum()
        weighted_loss.backward()
        clip_grad_norm_(self.phi_nn.parameters(), max_norm=10)
        self.phi_optim.step()
        
        return weighted_loss.detach().item()

    def learn(self, batch, **kwargs):
        # Update the target network if needed
        if self.update_count % self.target_update_freq == 0:
            self.sync_weight()
        
        if self.update_count % self.cycle_update_freq == 0:
            self.update_phi = not self.update_phi
        
        # Cyclical optimisation as recommended in the paper
        # Algorithm 1 does not suggest this though
        if self.update_phi:
            self.phi_l2_loss = self.phi_update(batch=batch)
        else:
            self.psi_td_loss = self.psi_update(batch=batch)

        # Increment the iteration counter
        self.update_count += 1
        
        stats = SFBertTrainingStats()
        stats.epsilon = self.eps
        stats.phi_l2_loss = self.phi_l2_loss
        stats.psi_td_loss = self.psi_td_loss
        stats.terminal_freq = self.terminal_freq

        return stats

    def process_fn(self, batch, buffer, indices):
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


if __name__ == '__main__':
    import os
    from utils import setup_artefact_paths, setup_experiment
    from yaml_utils import load_yaml
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    from tianshou.data import Collector, ReplayBuffer
    from tianshou.trainer import OffpolicyTrainer
    from algos.common import EpsilonDecayHookFactory, SaveHookFactory
    from envs.taxicab.language_taxicab import LanguageTaxicab, LanguageTaxicabFactory
    from algos.nets import precompute_bert_embeddings, extract_bert_layer_embeddings, FCMultiHead, FCTrunk
    
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

    env_factory = LanguageTaxicabFactory(
        hparams=env_hparams, 
        store_path=store_path
    )
    train_env: LanguageTaxicab = env_factory.get_env(set_id='TRAIN')
    test_env: LanguageTaxicab = env_factory.get_env(set_id='HOLDOUT')
    
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
    layer_embeddings = extract_bert_layer_embeddings(embedding_dict=precomp_embeddings, layer_ind=bert_layer_ind)
    
    in_dim = train_env.observation_space["features"].shape[0] + layer_embeddings[next(iter(layer_embeddings))].shape[0]
    
    rb = ReplayBuffer(size=exp_hparams['buffer_size'])
    
    phi_nn = FCTrunk(
        in_dim=train_env.observation_space["features"].shape[0],
        h=exp_hparams["phi_nn_dim"],
        device=device
    )

    psi_nn = FCMultiHead(
        in_dim=exp_hparams["phi_nn_dim"][-1],
        num_heads=train_env.action_space.n,
        h=exp_hparams["psi_nn_dim"],
        device=device
    )
    
    agent = SFBert(
        phi_nn=phi_nn, 
        psi_nn=psi_nn, 
        action_space=train_env.action_space,
        precomp_embeddings=layer_embeddings,
        lr=exp_hparams["step_size"],
        target_update_freq=exp_hparams["target_update_steps"], 
        gamma=env_hparams["disc_fact"],
        seed=seed,
        device=device
    )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
    test_collector = Collector(agent, test_env, exploration_noise=True)
    
    n_epochs = exp_hparams["n_epochs"]
    n_steps = exp_hparams["epoch_steps"]
    epoch_hook_factory = EpsilonDecayHookFactory(hparams=exp_hparams, max_steps=n_epochs*n_steps, agent=agent, logger=logger)
    save_hook_factory = SaveHookFactory(save_path=f'{store_path}/best_model.pth')
    
    result = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs, step_per_epoch=n_steps, step_per_collect=exp_hparams["step_per_collect"],
        update_per_step=exp_hparams["update_per_step"], episode_per_test=exp_hparams["episode_per_test"], batch_size=exp_hparams["batch_size"],
        train_fn=epoch_hook_factory.hook,
        test_fn=lambda epoch, global_step: agent.set_eps(exp_hparams["test_epsilon"]),
        save_best_fn=save_hook_factory.hook,
        logger=logger
    ).run()
    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')