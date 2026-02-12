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
from algos.nets import FCTree

@dataclass
class SFCondTrainingStats(TrainingStats):
    psi_td_loss: float = 0.0
    phi_l2_loss: float = 0.0
    rec_loss: float = 0.0
    epsilon: float = 0.0
    terminal_freq: float = 0.0
    norm_phi: float = 0.0
    norm_psi: float = 0.0


class SFCond(BasePolicy):
    def __init__(
            self, 
            phi_nn:torch.nn.Module, 
            psi_nn:torch.nn.Module, 
            precomp_embeddings:Dict,
            rb:ReplayBuffer,
            action_space, 
            phi_lr:float,
            psi_lr:float,
            phi_lambda:float,
            psi_lambda:float,
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
        self.phi_lambda = phi_lambda
        self.psi_lambda = psi_lambda
        self.gamma = gamma
        self.rb = rb

        # Psi(s,a) and Phi(s) share the same base. To obtain Psi(s,*) call phi_s = Phi(s) first
        # and then call Psi(phi_s). This gives Psi(s,a) for all a.
        self.phi_nn: nn.Module = phi_nn
        self.phi_nn_t: nn.Module = deepcopy(phi_nn)
        for p in self.phi_nn_t.parameters():
            p.requires_grad = False
        self.phi_nn_t.eval()
        self._sync_phi_target()

        self.psi_nn: nn.Module = psi_nn
        self.phi_optim = torch.optim.Adam(self.phi_nn.parameters(), lr=self.phi_lr)
        self.psi_optim = torch.optim.Adam(self.psi_nn.parameters(), lr=self.psi_lr)

        self.psi_nn_t: nn.Module = deepcopy(psi_nn)
        for p in self.psi_nn_t.parameters():
            p.requires_grad = False
        self.psi_nn_t.eval()
        self._sync_psi_target()

        self.update_phi = True
        
        self.dec_nn = dec_nn
        if self.dec_nn is not None:
            self.use_reconstruction_loss = True
            self.dec_optim = torch.optim.Adam(self.dec_nn.parameters(), lr=self.phi_lr)
        else:
            self.use_reconstruction_loss = False
            self.dec_optim = None

        self.precomp_embed = precomp_embeddings
        for key in self.precomp_embed.keys():
            self.precomp_embed[key] = self.precomp_embed[key] / (self.precomp_embed[key].norm(p=2) + 1e-8)
            self.precomp_embed[key] = self.precomp_embed[key].to(self.device)

        self.phi_update_tau = phi_update_tau
        self.psi_update_tau = psi_update_tau
        self.phi_update_freq = phi_update_ratio

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

    def _sync_psi_target(self) -> None:
        """Synchronize the weight for the target network."""
        self.psi_nn_t.load_state_dict(self.psi_nn.state_dict())
    
    def _sync_phi_target(self) -> None:
        """Synchronize the weight for the target network."""
        self.phi_nn_t.load_state_dict(self.phi_nn.state_dict())

    def _soft_update_psi_target(self):
        with torch.no_grad():
            for p_t, p in zip(self.psi_nn_t.parameters(), self.psi_nn.parameters()):
                p_t.data.mul_(1 - self.psi_update_tau).add_(self.psi_update_tau * p.data)
    
    def _soft_update_phi_target(self):
        with torch.no_grad():
            for p_t, p in zip(self.phi_nn_t.parameters(), self.phi_nn.parameters()):
                p_t.data.mul_(1 - self.phi_update_tau).add_(self.phi_update_tau * p.data)
    
    def instr_to_embedding(self, instrs) -> torch.Tensor:
        w = torch.stack(
                [self.precomp_embed[instr] for instr in instrs]
            ).to(self.device)
        return w

    def forward(self, batch, state=None, **kwargs):
        with torch.no_grad():
            numerical_features = batch.obs.features
            # Retrieve embeddings for instructions
            # shape: (batch_dim, embedding_dim)
            w = self.instr_to_embedding(instrs=batch.obs.instr)
            
            # shape: (batch_dim, num_actions, embedding_dim)
            phi_sa = self.phi_nn(numerical_features)
            batch_dim, num_actions, embedding_dim = phi_sa.shape

            # shape: (batch_dim, num_actions, embedding_dim)
            w_rep = w.unsqueeze(1).expand(batch_dim, num_actions, embedding_dim)
            # shape: (batch_dim, num_actions, embedding_dim + embedding_dim)
            x = torch.cat([phi_sa, w_rep], dim=-1)
            # shape: (batch_dim, num_actions, embedding_dim)
            psi = self.psi_nn(x.reshape(batch_dim * num_actions, 2 * embedding_dim)).reshape(batch_dim, num_actions, -1)
            
            # shape: (batch_dim, num_actions)
            q_logits = (psi * w.unsqueeze(1)).sum(dim=-1)
            dist = Categorical(logits=q_logits)
            act = argmax_random_tiebreak(q_logits)
            # act = dist.sample()
        return Batch(act=act, state=state, dist=dist)
    
    def psi_update(self, batch: Batch) -> float:
        batch_dim = len(batch)
        
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

        # Get the active instruction when the transition was played
        w = self.instr_to_embedding(instrs=batch.obs.instr)
        
        # Get phi(s,a) for each action a \in A for a given state s
        with torch.no_grad():
            phi_sa = self.phi_nn_t(batch.obs.features)
            phis_selected = phi_sa[torch.arange(batch_dim), acts_selected, :]
        batch_dim, num_actions, embedding_dim = phi_sa.shape

        # shape: (batch_dim, num_actions, embedding_dim)
        w_rep = w.unsqueeze(1).expand(batch_dim, num_actions, embedding_dim)
        # shape: (batch_dim, num_actions, embedding_dim + embedding_dim)
        x = torch.cat([phi_sa, w_rep], dim=-1)
        # shape: (batch_dim, num_actions, embedding_dim)
        psis = self.psi_nn(x.reshape(batch_dim * num_actions, 2 * embedding_dim)).reshape(batch_dim, num_actions, -1)
        # Psi outputs need to be converted to Q-values to get the optimal action to index Psi outputs
        # shape (batch_size, embedding_dim)
        psis_selected = psis[torch.arange(batch_dim), acts_selected, :]

        # Get the relevant Psi(s',a') vector for the greedy action a' to be played in the transition next_state. 
        with torch.no_grad():
            phi_sa_next = self.phi_nn_t(batch.obs_next.features)
            
            x_next = torch.cat([phi_sa_next, w_rep], dim=-1)
            psis_next = self.psi_nn_t(x_next.reshape(batch_dim * num_actions, 2 * embedding_dim)).reshape(batch_dim, num_actions, -1)
            
            qs_next = (psis_next * w.unsqueeze(1)).sum(dim=-1)
            acts_greedy = argmax_random_tiebreak(qs=qs_next)
            psis_next_greedy = psis_next[torch.arange(batch_dim), acts_greedy, :]
            
            # Get Phi(s) (equivalent to the reward in the standard Bellman update)
            psis_target = phis_selected + (1. - done).unsqueeze(-1) * self.gamma * psis_next_greedy

        # For debugging purposes
        phi_norm = phis_selected.norm(dim=-1).mean().item()
        psi_tgt_norm = psis_target.norm(dim=-1).mean().item()
        
        self.psi_optim.zero_grad()
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
        clip_grad_norm_(self.psi_nn.parameters(), max_norm=10)
        self.psi_optim.step()
        return loss.detach().cpu().numpy(), phi_norm, psi_tgt_norm
    
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
        
        return reward_loss.detach().item(), rec_loss.detach().item()

    def learn(self, batch, **kwargs):
        
        # Update phi from time to time
        if self.update_count % int(1/self.phi_update_freq) == 0:
            self.phi_l2_loss, self.rec_loss = self.phi_update(batch=batch)
            # print("Phi updated")
        
        # Update psi on every pass
        td_error, phi_norm, psi_norm = self.psi_update(batch=batch)
        psi_td_loss = td_error.mean()
        if hasattr(batch, "weight"):
            self.rb.update_weight(index=batch.indices, new_weight=td_error)
        self._soft_update_psi_target()
        # print("Psi updated")
        # Increment the iteration counter
        self.update_count += 1

        stats = SFCondTrainingStats()
        stats.epsilon = self.eps
        stats.phi_l2_loss = self.phi_l2_loss
        stats.rec_loss = self.rec_loss

        stats.psi_td_loss = psi_td_loss
        stats.norm_phi = phi_norm
        stats.norm_psi = psi_norm

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
    from envs.shapes.multitask_shapes import MultitaskShapes, ShapesPositionFactory
    from algos.embedding_ops import precompute_bert_embeddings, extract_bert_layer_embeddings
    from algos.nets import FCTree, FCTrunk
    
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

    phi_nn = FCTree(
        in_dim=train_env.observation_space["features"].shape,
        num_heads=train_env.action_space.n,
        h_trunk=exp_hparams["phi_trunk_dim"],
        h_head=exp_hparams["phi_head_dim"],
        device=device
    )

    psi_nn = FCTrunk(
        in_dim=2*exp_hparams["phi_head_dim"][-1] if isinstance(exp_hparams["phi_head_dim"], list) else exp_hparams["phi_head_dim"],
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
    
    agent = SFCond(
        phi_nn=phi_nn, 
        psi_nn=psi_nn,
        dec_nn=dec_nn,
        rb=rb,
        action_space=train_env.action_space,
        precomp_embeddings=layer_embeddings,
        l2_freq_scaling=exp_hparams["l2_freq_scaling"],
        phi_lr=exp_hparams["phi_lr"],
        psi_lr=exp_hparams["psi_lr"],
        phi_lambda=exp_hparams["phi_lambda"],
        psi_lambda=exp_hparams["psi_lambda"],
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