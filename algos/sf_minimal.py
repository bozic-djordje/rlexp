import torch
from typing import Any, Dict, Optional, Tuple
from nets import FCTree, LinearRegression
    

class SFMinimal:
    def __init__(
            self, 
            action_space,
            w_lr:float,
            psi_lr:float,
            lam:float,
            num_skills:int,
            h_trunk:Tuple[int],
            h_head:Tuple[int],
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
        self.psi_net = FCTree(
                in_dim=self.ftr_dim,
                num_heads=self.num_skills,
                h_trunk=h_trunk,
                h_head=h_head,
                non_linear=True,
                device=self.device
            )
        self.psi_optim: Dict[str, torch.optim.Adam] = {}
        self.instr_map: Dict[str, int] = {}

        # Dict of linear regression nn.Modules, indexed by task (instruction).
        self.r_hat: Dict[LinearRegression] = dict()
        self.r_optim: Dict = {}

        # To be set by trainer
        self.eps = None
        self.rng = torch.Generator().manual_seed(seed)
    
    def _sa(self, s, a):
        a_oh = torch.zeros(self.max_a_num, device=self.device)
        a_oh[int(a)] = 1.0
        return torch.cat([s, a_oh], dim=0)

    def _sa_batch(self, s):
        s = torch.as_tensor(s, device=self.device).float()
        if s.dim() == 1 or (s.dim() == 2 and s.shape[0] == 1):
            s = s.flatten()
            a_eye = torch.eye(self.max_a_num, device=self.device)
            s_rep = s.unsqueeze(0).repeat(self.max_a_num, 1)
            return torch.cat([s_rep, a_eye], dim=1)

        if s.dim() > 2:
            s = s.view(s.size(0), -1)
        bsz = s.shape[0]
        a_eye = torch.eye(self.max_a_num, device=self.device).unsqueeze(1).repeat(1, bsz, 1)
        s_rep = s.unsqueeze(0).repeat(self.max_a_num, 1, 1)
        return torch.cat([s_rep, a_eye], dim=2)

    def _init_task(self, instr):
        if instr in self.instr_map:
            return
        
        task_idx = len(self.instr_map)
        self.instr_map[instr] = task_idx
        
        # If we had a previous task, we initialise new task with those parameters
        if task_idx > 0:
            prev_task_idx = task_idx - 1
            self.psi_net.multihead.heads[task_idx].load_state_dict(
                self.psi_net.multihead.heads[prev_task_idx].state_dict()
            )
        
        # We have one optimiser per task, handling shared parameters as well
        psi_params = list(self.psi_net.trunk.parameters())
        psi_params += list(self.psi_net.multihead.heads[task_idx].parameters())
        self.psi_optim[instr] = torch.optim.Adam(psi_params, lr=self.psi_lr)
        
        self.r_hat[instr] = LinearRegression(in_dim=self.ftr_dim, device=self.device)
        self.r_optim[instr] = torch.optim.Adam(self.r_hat[instr].parameters(), lr=self.w_lr)
    
    def set_eps(self, eps: float):
        self.eps = eps

    def forward(self, s):
        instr = s["instr"]
        s_vec = torch.as_tensor(s["features"], device=self.device).flatten().float()

        self._init_task(instr=instr)

        eps = 0.0 if self.eps is None else self.eps
        if torch.rand((), generator=self.rng).item() < eps:
            return int(torch.randint(0, self.max_a_num, (1,), generator=self.rng).item())

        w = self.r_hat[instr].weight.flatten()
        sa_batch = self._sa_batch(s_vec)
        psi_all = self.psi_net(sa_batch)
        
        # psi_all: (A, N, D), w: (D,) -> q_all: (A, N)
        q_all = torch.matmul(psi_all, w)
        # We only consult q-values of previously encountered tasks
        q_all = q_all[:, :len(self.instr_map)]
        q_best = q_all.max(dim=1).values
        return int(torch.argmax(q_best).item())
    
    def _to_batched_tensor(self, x, dtype, expand_dim:bool=True) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).to(dtype)
        if expand_dim and x.dim() == 1:
            x = x.unsqueeze(0)
        return x
    
    def psi_update(self, batch) -> float:
        instrs = batch.obs["instr"]
        instr = instrs[0] if hasattr(instrs, "__len__") and not isinstance(instrs, str) else instrs
        self._init_task(instr=instr)
        task_idx = self.instr_map[instr]

        s_batch = self._to_batched_tensor(x=batch.obs["features"], dtype=torch.float32)
        s_nxt_batch = self._to_batched_tensor(x=batch.obs_next["features"], dtype=torch.float32)
        a_batch = self._to_batched_tensor(x=batch.act, dtype=torch.int64, expand_dim=False)
        
        a_onehot = torch.zeros(s_batch.size(0), self.max_a_num, device=self.device)
        a_onehot.scatter_(1, a_batch.unsqueeze(1), 1.0)
        sa = torch.cat([s_batch, a_onehot], dim=1)

        done = self._to_batched_tensor(x=batch.terminated, dtype=torch.bool, expand_dim=False)
        if hasattr(batch, "truncated"):
            done = done | self._to_batched_tensor(x=batch.truncated, dtype=torch.bool, expand_dim=False)
        done = done.float()

        w = self.r_hat[instr].weight.flatten()
        
        with torch.no_grad():
            sa_nxt_batch = self._sa_batch(s_nxt_batch)
            if sa_nxt_batch.dim() == 2:
                psi_nxt_all = self.psi_net(sa_nxt_batch).unsqueeze(1)
            else:
                a_num, batch_dim, _ = sa_nxt_batch.shape
                psi_nxt_all = self.psi_net(sa_nxt_batch.reshape(a_num * batch_dim, -1))
                psi_nxt_all = psi_nxt_all.view(a_num, batch_dim, self.num_skills, -1)

            # psi_nxt_all: (A, B, N, D), w: (D,) -> q_nxt_all: (A, B, N)
            q_nxt_all = torch.matmul(psi_nxt_all, w)
            # keep only seen skills: (A, B, N_seen)
            q_nxt_all = q_nxt_all[:, :, :len(self.instr_map)]
            # max over skills -> (A, B)
            q_nxt = q_nxt_all.max(dim=2).values
            # greedy action per batch element -> (B,)
            a_greedy = torch.argmax(q_nxt, dim=0)
            batch_idx = torch.arange(q_nxt.shape[1], device=self.device)
            psi_nxt = psi_nxt_all[a_greedy, batch_idx, task_idx]
            if psi_nxt.dim() == 1:
                psi_nxt = psi_nxt.unsqueeze(0)

        psi_pred = self.psi_net(sa)[:, task_idx, :]
        target = sa + (1 - done).unsqueeze(1) * self.gamma * psi_nxt
        td = target - psi_pred

        loss = (td * td).mean()
        opt = self.psi_optim[instr]
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def w_update(self, batch) -> float:
        instrs = batch.obs["instr"]
        instr = instrs[0] if hasattr(instrs, "__len__") and not isinstance(instrs, str) else instrs
        self._init_task(instr=instr)

        s_batch = self._to_batched_tensor(x=batch.obs["features"], dtype=torch.float32)
        a_batch = self._to_batched_tensor(x=batch.act, dtype=torch.int64, expand_dim=False)
        
        a_onehot = torch.zeros(s_batch.size(0), self.max_a_num, device=self.device)
        a_onehot.scatter_(1, a_batch.unsqueeze(1), 1.0)
        sa = torch.cat([s_batch, a_onehot], dim=1)

        r_t = self._to_batched_tensor(x=batch.rew, dtype=torch.float32)
        
        pred = self.r_hat[instr](sa).squeeze(-1)
        loss = (pred - r_t) ** 2
        loss = loss.mean()
        if self.lam is not None:
            W = self.r_hat[instr].linear.weight
            loss = loss + self.lam * torch.abs(W).sum()

        opt = self.r_optim[instr]
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return float(loss.item())
    
    def update(self, batch) -> Tuple:
        psi_loss = self.psi_update(batch=batch)
        w_loss = self.w_update(batch=batch)
        return psi_loss, w_loss
    
    def state_dict(self) -> Dict[str, Any]:
        r_hat_state = {}
        for instr, model in self.r_hat.items():
            r_hat_state[instr] = {
                "in_dim": model.in_dim,
                "state_dict": model.state_dict(),
            }

        r_optim_state = {}
        for instr, opt in self.r_optim.items():
            r_optim_state[instr] = opt.state_dict()

        psi_optim_state = {}
        for instr, opt in self.psi_optim.items():
            psi_optim_state[instr] = opt.state_dict()

        return {
            "w_lr": self.w_lr,
            "psi_lr": self.psi_lr,
            "lam": self.lam,
            "gamma": self.gamma,
            "eps": self.eps,
            "rng_state": self.rng.get_state(),
            "psi_state": self.psi_net.state_dict(),
            "psi_optim": psi_optim_state,
            "instr_map": self.instr_map,
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

        psi_state = state.get("psi_state")
        self.psi_net.load_state_dict(psi_state)

        self.psi_optim = {}
        for instr, opt_state in state.get("psi_optim", {}).items():
            if instr not in self.instr_map:
                continue
            head_idx = self.instr_map[instr]
            psi_params = list(self.psi_net.trunk.parameters())
            psi_params += list(self.psi_net.multihead.heads[head_idx].parameters())
            opt = torch.optim.Adam(psi_params, lr=self.psi_lr)
            opt.load_state_dict(opt_state)
            self.psi_optim[instr] = opt

        self.r_hat = {}
        self.r_optim = {}
        for instr, blob in state["r_hat"].items():
            model = LinearRegression(
                in_dim=blob["in_dim"],
                device=self.device,
            )
            model.load_state_dict(blob["state_dict"])
            self.r_hat[instr] = model

            opt = torch.optim.Adam(model.parameters(), lr=self.w_lr)
            opt.load_state_dict(state["r_optim"][instr])
            self.r_optim[instr] = opt

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
    
    agent = SFMinimal(
        action_space=env.action_space,
        w_lr=exp_hparams["w_lr"],
        psi_lr=exp_hparams["psi_lr"],
        lam=exp_hparams["lambda"],
        num_skills=num_tasks,
        s_dim=env.obs["features"].shape[0],
        h_trunk=tuple(h_trunk),
        h_head=tuple(h_head),
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

    total_episodes = num_tasks * env_hparams["goal_resample_t"]
    with tqdm(total=total_episodes, desc="Episodes", unit="ep") as pbar:
        while not tasks_done:
            a = agent.forward(s=s)
            s_next, r, is_terminal, truncated, info = env.step(action=a)
            done = is_terminal or truncated
            
            global_step += 1
            task_step += 1
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

            if len(rb) >= batch_size:
                batch, _ = rb.sample(batch_size=batch_size)
                psi_loss, w_loss = agent.update(batch=batch)
            else:
                psi_loss, w_loss = 0.0, 0.0
            epoch_hook.hook(epoch=None, global_step=task_step, logging_step=global_step)

            logger.write(
                "train/epoch",
                global_step,
                {
                    "psi_loss": psi_loss,
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
