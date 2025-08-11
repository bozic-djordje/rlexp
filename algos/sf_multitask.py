import time
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from typing import Dict, List, Optional, Tuple
from tianshou.data.buffer.base import Batch
from tianshou.data import ReplayBuffer
from tianshou.policy.base import TrainingStats
from tianshou.policy import BasePolicy
from torch.distributions import Categorical
from tianshou.utils.torch_utils import torch_train_mode
from algos.common import BetaAnnealHook, CompositeHook, argmax_random_tiebreak
from algos.common import GroupedReplayBuffer


@dataclass
class SFBaseTrainingStats(TrainingStats):
    psi_td_loss: float = 0.0
    phi_l2_loss: float = 0.0
    rec_loss: float = 0.0
    epsilon: float = 0.0
    terminal_freq: float = 0.0

# TODO: How does knowledge transfer occurr here exactly? When we encounter a new goal, shouldn't we copy the
# current parameters of the closest model we currently have?
class MultiparamModule:
    def __init__(self, base_module: nn.Module, num_modules: int, lr: float):
        self.num_modules = num_modules
        self._counters: List[int] = [0 for _ in range(num_modules)]

        self._modules: List[nn.Module] = [deepcopy(base_module) for _ in range(num_modules)]
        self._t_modules: List[nn.Module] = [deepcopy(base_module) for _ in range(num_modules)]

        for index, m in enumerate(self._t_modules):
            m.load_state_dict(self._modules[index].state_dict())
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        self._optims = [torch.optim.Adam(m.parameters(), lr=lr) for m in self._modules]
        
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
            return self._t_modules[self._key_map[key]]
        else:
            return self._modules[self._key_map[key]]
    
    def get_optim(self, key:str) -> torch.optim.Optimizer:
        self._check_add_key(key=key)
        return self._optims[self._key_map[key]]
    
    def get_counter(self, key:str) -> int:
        self._check_add_key(key=key)
        return self._counters[self._key_map[key]]
    
    def update_counter(self, key:str) -> int:
        self._check_add_key(key=key)
        self._counters[self._key_map[key]] += 1
    
    def sync_module_weights(self, key:str) -> None:
        t_module = self.get_module(key=key, target=True)
        module = self.get_module(key=key, target=False)
        t_module.load_state_dict(module.state_dict())

    def forward(self, x: torch.Tensor, state: Dict=None, **kwargs):
        if "key" in state:
            module = self.get_module(key=state["key"])
            result = module(x, **kwargs)
        else:
            result = [module(x, **kwargs) for module in self._modules]
        return result
    
    def target(self, x: torch.Tensor, state: Dict=None, **kwargs):
        if "key" in state:
            module = self.get_module(key=state["key"], target=True)
            result = module(x, **kwargs)
        else:
            result = [module(x, **kwargs) for module in self._t_modules]
        return result
    

class SFBase(BasePolicy):
    def __init__(
            self, 
            phi_nn:torch.nn.Module, 
            psi_nn:torch.nn.Module, 
            precomp_embeddings:Dict,
            rb:ReplayBuffer,
            num_skills,
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
        self.phi_optim = torch.optim.Adam(self.phi_nn.parameters(), lr=self.lr)
        
        self.psi_nn: MultiparamModule = MultiparamModule(
            base_module=psi_nn, 
            num_modules=num_skills, 
            lr=self.lr
        )
        
        self.update_phi = True
        self.current_w = None
        
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

    # def sync_weight(self) -> None:
    #     """Synchronize the weight for the target network."""
    #     self.psi_nn_t.load_state_dict(self.psi_nn.state_dict())
    
    def instr_to_embedding(self, instrs) -> torch.Tensor:
        w = torch.stack(
                [self.precomp_embed[instr] for instr in instrs]
            ).to(self.device)
        return w
    
    @staticmethod
    def embedding_to_key(embedding: torch.Tensor) -> Tuple:
        return tuple(embedding.detach().tolist())

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
                st = {"key": self.embedding_to_key(w[0])}
                psi_a = [self.psi_nn.forward(chunk.squeeze(1), st).unsqueeze(1) for chunk in phi_a]
            else:
                # If elements belong to different instructions, we need to use different parameterisations for psi network. 
                # This means we need to parse each element separately, hence the loop below.
                psi_a = []
                for index in range(len(batch)):
                    transition = batch[index]
                    key = self.embedding_to_key(w[index])
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
            # TODO: Double check if this will work.
            self.current_w = str(w[0]).item()
            
            # shape: (batch_dim, num_actions)
            q_logits = torch.bmm(psi, w.unsqueeze(2)).squeeze(2)
            dist = Categorical(logits=q_logits)
            act = argmax_random_tiebreak(q_logits)
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
        key = self.embedding_to_key(embedding=w[0])
        st = {"key": key}
        
        # Get phi(s,a) for each action a \in A for a given state s
        with torch.no_grad():
            phi_sa = self.phi_nn(batch.obs.features)
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
            
            phi_sa_next = self.phi_nn(batch.obs_next.features)
            phi_next_a = torch.chunk(phi_sa_next, chunks=num_actions, dim=1)
            psis_next = [self.psi_nn.target(chunk.squeeze(1), st).unsqueeze(1) for chunk in phi_next_a]
            psis_next = torch.cat(psis_next, dim=1)
            
            qs_next = torch.bmm(psis_next, w.unsqueeze(2)).squeeze(2)
            acts_greedy = argmax_random_tiebreak(qs=qs_next)
            psis_next_greedy = psis_next[torch.arange(batch_size), acts_greedy, :]
            
            # Get Phi(s) (equivalent to the reward in the standard Bellman update)
            psis_target = phis_selected + (1. - terminated).unsqueeze(-1) * self.gamma * psis_next_greedy

        self.psi_nn.get_optim(key=key).zero_grad()
        td_error = (psis_selected - psis_target).pow(2).sum(dim=1)
        loss = (torch.tensor(batch.weight).to(self.device) * td_error).mean() if hasattr(batch, "weight") else td_error.mean()
        loss.backward()
        clip_grad_norm_(self.psi_nn.get_module(key=key).parameters(), max_norm=10)
        self.psi_nn.get_optim(key=key).step()
        return td_error.detach().cpu().numpy()
    
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
    
    def update(self, sample_size, buffer: GroupedReplayBuffer, **kwargs):
        start_time = time.time()
       
        phi_batch, phi_indices = buffer.sample(sample_size)
        phi_batch = self.process_fn(phi_batch, buffer, phi_indices)

        # All samples in a batch need to have the same instruction
        # so they could be processed as a batch
        psi_batch, psi_indices = buffer.sample_group(
            group_id=self.current_w, 
            batch_size=sample_size
        )
        psi_batch = self.process_fn(psi_batch, buffer, psi_indices)
        self.updating = True
        
        with torch_train_mode(self):
            training_stat = self.learn(phi_batch, psi_batch, **kwargs)
        
        self.post_process_fn(phi_batch, buffer, phi_indices)
        self.post_process_fn(psi_batch, buffer, psi_indices)
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        
        training_stat.train_time = time.time() - start_time
        
        return training_stat

    def learn(self, phi_batch, psi_batch, **kwargs):
        # Increment the iteration counter
        self.update_count += 1
        self.psi_nn.update_counter(key=self.current_w)
        
        # Update the target network if needed
        if self.psi_nn.get_counter(key=self.current_w) % self.target_update_freq == 0:
            self.psi_nn.sync_module_weights(key=self.current_w)
        
        # Cyclical optimisation adapted for our needs
        # We update both phi and psi, but using different batches. 
        # This is equivalent to alternating optimisation with interval = 1
        self.phi_l2_loss, self.rec_loss = self.phi_update(batch=phi_batch)
        td_error = self.psi_update(batch=psi_batch)
        self.psi_td_loss = td_error.mean()
        
        # TODO: Prioritised replay not supported yet
        # if hasattr(phi_batch, "weight"):
        #     self.rb.update_weight(index=phi_batch.indices, new_weight=td_error)
        
        stats = SFBaseTrainingStats()
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
    from algos.common import EpsilonDecayHook, SaveHook, SFTrainer, GroupedReplayBuffer
    from envs.shapes.multitask_shapes import MultitaskShapes, ShapesPositionFactory
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
    
    rb = GroupedReplayBuffer(size=exp_hparams['buffer_size'])

    phi_nn = FCMultiHead(
        in_dim=train_env.observation_space["features"].shape[0],
        num_heads=train_env.action_space.n,
        h=exp_hparams["phi_nn_dim"],
        device=device
    )

    psi_nn = FCTrunk(
        in_dim=exp_hparams["phi_nn_dim"][-1],
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
    
    agent = SFBase(
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