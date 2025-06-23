from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from typing import Any, Dict, Optional
from tianshou.data.buffer.base import Batch
from tianshou.data import ReplayBuffer, PrioritizedReplayBuffer
from tianshou.policy.base import TrainingStats
from tianshou.policy import BasePolicy
from torch.distributions import Categorical
from algos.common import BetaAnnealHook, CompositeHook, argmax_random_tiebreak

@dataclass
class SFBertTrainingStats(TrainingStats):
    psi_td_loss: float = 0.0
    phi_l2_loss: float = 0.0
    rec_loss: float = 0.0
    epsilon: float = 0.0
    terminal_freq: float = 0.0


class SFBert(BasePolicy):
    def __init__(
            self, 
            phi_nn:torch.nn.Module, 
            psi_nn:torch.nn.Module, 
            precomp_embeddings:Dict,
            rb:ReplayBuffer,
            action_space, 
            lr:float, 
            target_update_freq:int, 
            cycle_update_freq:int, 
            l2_freq_scaling:bool, 
            gamma:float=0.99, 
            seed:float=1., 
            terminal_rew:float=20,
            dec_nn:Optional[torch.nn.Module]=None, 
            device:torch.device=torch.device("cpu")
        ):
        super().__init__(action_space=action_space)
        self.device = device

        self.lr = lr
        self.gamma = gamma
        self.rb = rb

        # Psi(s,a) and Phi(s) share the same base. To obtain Psi(s,*) call phi_s = Phi(s) first
        # and then call Psi(phi_s). This gives Psi(s,a) for all a.
        self.phi_nn: nn.Module = phi_nn
        self.psi_nn: nn.Module = psi_nn
        self.phi_optim = torch.optim.Adam(self.phi_nn.parameters(), lr=self.lr)
        self.psi_optim = torch.optim.Adam(self.psi_nn.parameters(), lr=self.lr)

        self.psi_nn_t: nn.Module = deepcopy(psi_nn)
        for p in self.psi_nn_t.parameters():
            p.requires_grad = False
        self.psi_nn_t.eval()
        self.sync_weight()
        
        self.update_phi = True
        
        self.dec_nn = dec_nn
        if self.dec_nn is not None:
            self.use_reconstruction_loss = True
            self.dec_optim = torch.optim.Adam(self.dec_nn.parameters(), lr=self.lr)
        else:
            self.use_reconstruction_loss = False
            self.dec_optim = None

        self.precomp_embed = precomp_embeddings
        for key in self.precomp_embed:
            self.precomp_embed[key] = self.precomp_embed[key].to(self.device)

        self.target_update_freq = target_update_freq
        self.cycle_update_freq = cycle_update_freq
        self.update_count = 0  # Counter for training iterations
        self.r_counter = Counter()
        self.l2_freq_scaling = l2_freq_scaling
        
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
            # act = argmax_random_tiebreak(q_logits)
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

        # We update the "live" network, self.current. First we zero out the optimizer gradients
        # and then we apply the update step using qs_selected and qs_target.
        self.psi_optim.zero_grad()
        td_error = (psis_selected - psis_target).pow(2).sum(dim=1)
        loss = (torch.tensor(batch.weight).to(self.device) * td_error).mean() if hasattr(batch, "weight") else td_error.mean()
        loss.backward()
        clip_grad_norm_(self.psi_nn.parameters(), max_norm=10)
        self.psi_optim.step()
        return td_error.detach().cpu().numpy()
    
    def phi_update(self, batch:Batch) -> float:
        if not isinstance(batch.rew, torch.Tensor):
            r_target = torch.tensor(batch.rew, dtype=torch.float32).to(self.device)
        else:
            r_target = batch.rew

        if not isinstance(batch.rew, torch.Tensor):
            obs_target = torch.tensor(batch.obs.features, dtype=torch.float32).to(self.device)
        else:
            obs_target = batch.obs.features
        
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

        phis = self.phi_nn(batch.obs.features)
        r_pred = torch.bmm(phis.unsqueeze(1), w.unsqueeze(2)).squeeze()

        reward_errors = (r_pred - r_target).pow(2)
        reward_loss = (weights * reward_errors).sum()

        if self.use_reconstruction_loss:
            s_hats = self.dec_nn(phis)
            rec_loss = (s_hats - obs_target).pow(2).mean()
            total_loss = reward_loss + rec_loss
        else:
            rec_loss = torch.tensor(0., device=self.device)
            total_loss = reward_loss

        total_loss.backward()

        # clip
        clip_grad_norm_(self.phi_nn.parameters(), 10)
        if self.use_reconstruction_loss:
            clip_grad_norm_(self.dec_nn.parameters(), 10)

        # step
        self.phi_optim.step()
        if self.use_reconstruction_loss:
            self.dec_optim.step()
        
        return reward_loss.detach().item(), rec_loss.detach().item()

    def learn(self, batch, **kwargs):
        # Increment the iteration counter
        self.update_count += 1
        
        # Update the target network if needed
        if self.update_count % self.target_update_freq == 0:
            self.sync_weight()
        
        if self.update_count % self.cycle_update_freq == 0:
            self.update_phi = not self.update_phi
        
        # Cyclical optimisation as recommended in the paper
        # Algorithm 1 does not suggest this though
        if self.update_phi:
            self.phi_l2_loss, self.rec_loss = self.phi_update(batch=batch)
        else:
            td_error = self.psi_update(batch=batch)
            self.psi_td_loss = td_error.mean()
            if hasattr(batch, "weight"):
                self.rb.update_weight(index=batch.indices, new_weight=td_error)
        
        stats = SFBertTrainingStats()
        stats.epsilon = self.eps
        stats.phi_l2_loss = self.phi_l2_loss
        stats.psi_td_loss = self.psi_td_loss
        stats.terminal_freq = self.terminal_freq
        stats.rec_loss = self.rec_loss

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


if __name__ == '__main__':
    import os
    from utils import setup_artefact_paths, setup_experiment
    from yaml_utils import load_yaml
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    from tianshou.data import Collector, ReplayBuffer
    from tianshou.trainer import OffpolicyTrainer
    from algos.common import EpsilonDecayHook, SaveHook
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
    
    if exp_hparams['prioritised_replay'] is False:
        rb = ReplayBuffer(size=exp_hparams['buffer_size'])
    else:
        rb = PrioritizedReplayBuffer(
            size=exp_hparams['buffer_size'], 
            alpha=exp_hparams["priority_alpha"], 
            beta=exp_hparams["priority_beta_start"]
        )
    
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
    
    agent = SFBert(
        phi_nn=phi_nn, 
        psi_nn=psi_nn,
        dec_nn=dec_nn,
        rb=rb,
        action_space=train_env.action_space,
        precomp_embeddings=layer_embeddings,
        l2_freq_scaling=exp_hparams["l2_freq_scaling"],
        lr=exp_hparams["step_size"],
        target_update_freq=exp_hparams["target_update_steps"],
        cycle_update_freq=exp_hparams["cycle_update_steps"],
        gamma=env_hparams["disc_fact"],
        seed=seed,
        device=device
    )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
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
    
    result = OffpolicyTrainer(
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