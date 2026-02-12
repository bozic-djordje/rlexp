import torch
from typing import Any, Dict, List, Optional, Tuple
from nets import LinearRegression
    

class SFTabular:
    def __init__(
            self, 
            action_space,
            w_lr:float,
            psi_lr:float,
            lam:float,
            num_skills:int,
            s_dim: int,
            gamma:float=0.99, 
            seed:int=1, 
            device:torch.device=torch.device("cpu")
        ):
        self.num_skills = num_skills
        self.action_space = action_space
        self.device = device
        
        self.logger = logger
        self.epoch = 0

        self.w_lr = w_lr
        self.psi_lr = psi_lr
        self.gamma = gamma
        self.lam = lam

        self.max_a_num = self.action_space.n
        self.ftr_dim = s_dim + self.max_a_num
        self.psi_table = torch.zeros(
            (self.num_skills, 0, self.max_a_num, self.ftr_dim),
            device=self.device,
        )
        self.instr_map: Dict[str, int] = {}
        self.state_map: Dict[Tuple[float, ...], int] = {}
        self.action_map: Dict[Any, int] = {i: i for i in range(self.max_a_num)}

        # Linear regression nn.Modules, indexed by instruction index.
        self.r_hat: List[Optional[LinearRegression]] = [None] * self.num_skills
        self.r_optim: List[Optional[torch.optim.Optimizer]] = [None] * self.num_skills

        # To be set by trainer
        self.eps = None
        self.rng = torch.Generator().manual_seed(seed)
    
    def _sa(self, s, a):
        a_oh = torch.zeros(self.max_a_num, device=self.device)
        a_oh[int(a)] = 1.0
        return torch.cat([s, a_oh], dim=0)

    def _state_key(self, s) -> Tuple[float, ...]:
        s = torch.as_tensor(s, device="cpu").flatten()
        return tuple(s.tolist())

    def _state_idx(self, s) -> int:
        key = self._state_key(s)
        if key in self.state_map:
            return self.state_map[key]
        idx = len(self.state_map)
        self.state_map[key] = idx
        new_state = torch.zeros(
            (self.num_skills, 1, self.max_a_num, self.ftr_dim),
            device=self.device,
        )
        self.psi_table = torch.cat([self.psi_table, new_state], dim=1)
        return idx

    def _action_idx(self, a) -> int:
        if isinstance(a, torch.Tensor):
            a = int(a.item())
        if a in self.action_map:
            return self.action_map[a]
        idx = len(self.action_map)
        if idx >= self.max_a_num:
            raise ValueError(f"action_map exceeded: {idx + 1} > {self.max_a_num}")
        self.action_map[a] = idx
        return idx

    def _init_task(self, instr) -> int:
        if instr in self.instr_map:
            return self.instr_map[instr]
        
        task_idx = len(self.instr_map)
        if task_idx >= self.num_skills:
            raise ValueError(f"num_skills exceeded: {task_idx + 1} > {self.num_skills}")
        self.instr_map[instr] = task_idx
        
        # If we had a previous task, we initialise new task with those parameters
        if task_idx > 0 and self.psi_table.size(1) > 0:
            prev_task_idx = task_idx - 1
            self.psi_table[task_idx].copy_(self.psi_table[prev_task_idx])
        
        model = LinearRegression(in_dim=self.ftr_dim, device=self.device)
        self.r_hat[task_idx] = model
        self.r_optim[task_idx] = torch.optim.Adam(model.parameters(), lr=self.w_lr)
        return task_idx
    
    def set_eps(self, eps: float):
        self.eps = eps

    def forward(self, s):
        instr = s["instr"]
        s_vec = torch.as_tensor(s["features"], device=self.device).flatten().float()

        t_idx = self._init_task(instr=instr)

        eps = 0.0 if self.eps is None else self.eps
        if torch.rand((), generator=self.rng).item() < eps:
            return int(torch.randint(0, self.max_a_num, (1,), generator=self.rng).item())

        s_idx = self._state_idx(s_vec)
        w = self.r_hat[t_idx].weight.flatten()
        n_skills = len(self.instr_map)
        # psi_all: (N_seen, A, D), w: (D,) -> q_all: (N_seen, A)
        psi_all = self.psi_table[:n_skills, s_idx]
        q_all = torch.matmul(psi_all, w)
        # We only consult q-values of previously encountered tasks
        q_best = q_all.max(dim=0).values
        return int(torch.argmax(q_best).item())
    
    def _to_batched_tensor(self, x, dtype, expand_dim:bool=True) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).to(dtype)
        if expand_dim and x.dim() == 1:
            x = x.unsqueeze(0)
        return x
    
    def psi_update(self, batch) -> float:
        if len(batch) < 1:
            return 0.0

        gpi_loss = 0.0
        mnt_loss = 0.0
        mnt_n_updates = 0

        for transition in batch:
            instr_t = transition.obs["instr"]
            t_idx = self._init_task(instr=instr_t)

            s_vec = torch.as_tensor(transition.obs["features"], device=self.device).flatten().float()
            s_nxt_vec = torch.as_tensor(transition.obs_next["features"], device=self.device).flatten().float()
            s_idx = self._state_idx(s_vec)
            s_nxt_idx = self._state_idx(s_nxt_vec)
            a_idx = self._action_idx(transition.act)
            done = transition.terminated

            n_skills = len(self.instr_map)

            # Current task weights w_t
            w_t = self.r_hat[t_idx].weight.flatten()  # (d,)

            # ----- Compute c = argmax_{k<=t} max_b psi_k(s,b)^T w_t -----
            # psi_s_all: (n_skills, n_actions, d)
            psi_s_all = self.psi_table[:n_skills, s_idx]
            # q_s_all: (n_skills, n_actions)
            q_s_all = torch.matmul(psi_s_all, w_t)
            # best value per skill: (n_skills,)
            best_per_skill = q_s_all.max(dim=1).values
            c_idx = int(torch.argmax(best_per_skill).item())

            # ----- Compute a' for SF update using GPI at s' under w_t -----
            psi_sp_all = self.psi_table[:n_skills, s_nxt_idx]          # (n_skills, n_actions, d)
            q_sp_all = torch.matmul(psi_sp_all, w_t)                   # (n_skills, n_actions)
            q_sp_gpi = q_sp_all.max(dim=0).values                      # (n_actions,)
            a_gpi = int(torch.argmax(q_sp_gpi).item())                 # scalar
            # IMPORTANT: bootstrap through current task's SFs (psi_t), not the winner's SFs
            psi_boot_t = self.psi_table[t_idx, s_nxt_idx, a_gpi]       # (d,)

            # Immediate feature vector phi(s,a,s') (or phi(s,a))
            phi_sa = self._sa(s=s_vec, a=a_idx)  # (d,)

            # ----- Update current task SFs: psi_t -----
            target_t = phi_sa + (1 - int(done)) * self.gamma * psi_boot_t
            td_t = target_t - self.psi_table[t_idx, s_idx, a_idx]
            self.psi_table[t_idx, s_idx, a_idx] += self.psi_lr * td_t
            gpi_loss += float((torch.sqrt(td_t * td_t)).mean().item())

            # ----- Maintenance update: if c != t then update psi_c under w_c -----
            if c_idx != t_idx:
                w_c = self.r_hat[c_idx].weight.flatten()

                # a'_c = argmax_b psi_c(s',b)^T w_c
                psi_c_sp = self.psi_table[c_idx, s_nxt_idx]            # (n_actions, d)
                q_c_sp = torch.matmul(psi_c_sp, w_c)                   # (n_actions,)
                a_c = int(torch.argmax(q_c_sp).item())
                psi_boot_c = self.psi_table[c_idx, s_nxt_idx, a_c]     # (d,)

                target_c = phi_sa + (1 - int(done)) * self.gamma * psi_boot_c
                td_c = target_c - self.psi_table[c_idx, s_idx, a_idx]
                self.psi_table[c_idx, s_idx, a_idx] += self.psi_lr * td_c
                mnt_loss += float((torch.sqrt(td_c * td_c)).mean().item())
                mnt_n_updates += 1
        
        gpi_loss /= max(1, len(batch))
        mnt_loss /= max(1, mnt_n_updates)

        return gpi_loss, mnt_loss


    def w_update(self, batch) -> float:
        instrs = batch.obs["instr"]
        instr = instrs[0] if hasattr(instrs, "__len__") and not isinstance(instrs, str) else instrs
        t_idx = self._init_task(instr=instr)

        s_batch = self._to_batched_tensor(x=batch.obs["features"], dtype=torch.float32)
        a_batch = self._to_batched_tensor(x=batch.act, dtype=torch.int64, expand_dim=False).squeeze(-1)
        
        a_onehot = torch.zeros(s_batch.size(0), self.max_a_num, device=self.device)
        a_onehot.scatter_(1, a_batch.unsqueeze(1), 1.0)
        sa = torch.cat([s_batch, a_onehot], dim=1)

        r_t = self._to_batched_tensor(x=batch.rew, dtype=torch.float32)
        
        model = self.r_hat[t_idx]
        pred = model(sa).squeeze(-1)
        loss = (pred - r_t) ** 2
        loss = loss.mean()
        if self.lam is not None:
            W = model.linear.weight
            loss = loss + self.lam * torch.abs(W).sum()

        opt = self.r_optim[t_idx]
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return float(torch.sqrt(loss).item())
    
    def update(self, batch) -> Tuple:
        gpi_loss, mnt_loss = self.psi_update(batch=batch)
        w_loss = self.w_update(batch=batch)
        return gpi_loss, mnt_loss, w_loss
    
    def state_dict(self) -> Dict[str, Any]:
        r_hat_state: List[Optional[Dict[str, Any]]] = [None] * self.num_skills
        for idx, model in enumerate(self.r_hat):
            if model is None:
                continue
            r_hat_state[idx] = {
                "in_dim": model.in_dim,
                "state_dict": model.state_dict(),
            }

        r_optim_state: List[Optional[Dict[str, Any]]] = [None] * self.num_skills
        for idx, opt in enumerate(self.r_optim):
            if opt is None:
                continue
            r_optim_state[idx] = opt.state_dict()

        return {
            "w_lr": self.w_lr,
            "psi_lr": self.psi_lr,
            "lam": self.lam,
            "gamma": self.gamma,
            "eps": self.eps,
            "rng_state": self.rng.get_state(),
            "psi_table": self.psi_table,
            "instr_map": self.instr_map,
            "state_map": self.state_map,
            "action_map": self.action_map,
            "r_hat": r_hat_state,
            "r_optim": r_optim_state,
            "max_a_num": self.max_a_num,
            "ftr_dim": self.ftr_dim,
            "num_skills": self.num_skills,
        }
    
    def load_state_dict(self, state: dict) -> None:
        self.w_lr = state["w_lr"]
        self.psi_lr = state["psi_lr"]
        self.lam = state["lam"]
        self.gamma = state["gamma"]
        self.eps = state["eps"]

        if "num_skills" in state:
            self.num_skills = state["num_skills"]
        if "max_a_num" in state:
            self.max_a_num = state["max_a_num"]
        if "ftr_dim" in state:
            self.ftr_dim = state["ftr_dim"]

        self.rng = torch.Generator()
        self.rng.set_state(state["rng_state"])

        self.instr_map = state.get("instr_map", {})
        self.state_map = state.get("state_map", {})
        self.action_map = state.get("action_map", {i: i for i in range(self.max_a_num)})

        self.psi_table = state.get("psi_table")
        if self.psi_table is None:
            self.psi_table = torch.zeros(
                (self.num_skills, 0, self.max_a_num, self.ftr_dim),
                device=self.device,
            )
        else:
            self.psi_table = self.psi_table.to(self.device)

        self.r_hat = [None] * self.num_skills
        self.r_optim = [None] * self.num_skills

        r_hat_state = state.get("r_hat", [])
        r_optim_state = state.get("r_optim", [])
        if isinstance(r_hat_state, dict):
            for instr, blob in r_hat_state.items():
                idx = self.instr_map[instr]
                model = LinearRegression(
                    in_dim=blob["in_dim"],
                    device=self.device,
                )
                model.load_state_dict(blob["state_dict"])
                self.r_hat[idx] = model

                opt = torch.optim.Adam(model.parameters(), lr=self.w_lr)
                opt.load_state_dict(r_optim_state[instr])
                self.r_optim[idx] = opt
        else:
            for idx, blob in enumerate(r_hat_state):
                if blob is None:
                    continue
                model = LinearRegression(
                    in_dim=blob["in_dim"],
                    device=self.device,
                )
                model.load_state_dict(blob["state_dict"])
                self.r_hat[idx] = model

                opt = torch.optim.Adam(model.parameters(), lr=self.w_lr)
                opt.load_state_dict(r_optim_state[idx])
                self.r_optim[idx] = opt

if __name__ == '__main__':
    import os
    from tqdm import tqdm
    from utils import setup_artefact_paths, setup_experiment
    from yaml_utils import load_yaml
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.data import ReplayBuffer, Batch
    from tianshou.utils import TensorboardLogger
    from algos.common import EpsilonDecayHook
    from envs.shapes.multitask_shapes import MultitaskShapes, ShapesAttrCombFactory
    
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

    env_factory = ShapesAttrCombFactory(
        hparams=env_hparams, 
        store_path=store_path
    )
    env: MultitaskShapes = env_factory.get_env(set_id='TRAIN')
    all_instructions = env_factory.get_all_instructions(set_id="TRAIN")
    num_tasks = len(all_instructions)

    h_trunk = exp_hparams.get("psi_trunk_dim", (128,))
    h_head = exp_hparams.get("psi_head_dim", (64,))
    if not isinstance(h_trunk, (list, tuple)):
        h_trunk = (h_trunk,)
    if not isinstance(h_head, (list, tuple)):
        h_head = (h_head,)
    
    agent = SFTabular(
        action_space=env.action_space,
        w_lr=exp_hparams["w_lr"],
        psi_lr=exp_hparams["psi_lr"],
        lam=exp_hparams["lambda"],
        num_skills=num_tasks,
        s_dim=env.obs["features"].shape[0],
        gamma=env_hparams["disc_fact"],
        seed=seed,
        device=device
    )

    buffer_size = exp_hparams.get("buffer_size", 10000)
    batch_size = exp_hparams.get("batch_size", 32)
    rb_by_instr = {instr: ReplayBuffer(size=buffer_size) for instr in all_instructions}

    episode_max_steps = 40 if env_hparams["max_steps"] is None else env_hparams["max_steps"]
    task_steps = episode_max_steps*env_hparams["goal_resample_t"]
    epoch_hook = EpsilonDecayHook(hparams=exp_hparams, max_steps=task_steps, agent=agent, logger=logger)

    s = env.obs
    prev_instr = s["instr"]

    tasks_done = False
    done = False
    global_step = 0
    
    task_step = 0
    ret = 0
    warmup_steps = exp_hparams["warmup_steps"]

    total_episodes = num_tasks * env_hparams["goal_resample_t"]
    with tqdm(total=total_episodes, desc="Episodes", unit="ep") as pbar:
        while not tasks_done:
            a = agent.forward(s=s)
            s_next, r, is_terminal, truncated, info = env.step(action=a)
            done = is_terminal or truncated
            
            global_step += 1
            ret += r
            
            instr = s["instr"]
            if instr not in rb_by_instr:
                rb_by_instr[instr] = ReplayBuffer(size=buffer_size)
            rb = rb_by_instr[instr]
            
            transition = Batch(
                obs=s,
                act=a,
                obs_next=s_next,
                rew=r,
                terminated=is_terminal,
                truncated=truncated,
            )
            rb.add(transition)

            if global_step >= warmup_steps:
                batch, _ = rb.sample(batch_size=batch_size)
                gpi_loss, mnt_loss, w_loss = agent.update(batch=batch)
                task_step += 1
            else:
                gpi_loss, mnt_loss, w_loss = 0.0, 0.0, 0.0
            epoch_hook.hook(epoch=None, global_step=task_step, logging_step=global_step)

            logger.write(
                "train/epoch",
                global_step,
                {
                    "gpi_loss": gpi_loss,
                    "mnt_loss": mnt_loss,
                    "w_loss": w_loss,
                    "task_idx": len(agent.instr_map),
                }
            )   

            if done:
                pbar.update(1)
                torch.save(agent.state_dict(), f'{store_path}/last_model.pth')
                logger.write("train/epoch", global_step, {"return": ret})

                s, info = env.reset()
                tasks_done = info["tasks_exhausted"]
                ret = 0
                if s["instr"] != prev_instr:
                    task_step = 0
                    prev_instr = s["instr"]
                    epoch_hook = EpsilonDecayHook(hparams=exp_hparams, max_steps=task_steps, agent=agent, logger=logger)
            else:
                s = s_next
