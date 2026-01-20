from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Set, Tuple, Union
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import re
from copy import deepcopy
from envs.shapes.shapes import Shapes, ShapesGoto, ShapesUnlock, ShapesPickup, ShapesRetrieve, DEFAULT_OBJECTS, Shape, Door


def generate_instruction(instr: str, goal: Union[Door|Shape]) -> str:
        # Split the sentence into words and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', instr)

        result_tokens = []
        for token in tokens:
            if hasattr(goal, token):
                result_tokens.append(getattr(goal, token))
            else:
                result_tokens.append(token)

        # Reconstruct sentence with spacing
        sentence = ''
        for i, tok in enumerate(result_tokens):
            if i > 0 and re.match(r'\w', tok) and re.match(r'\w', result_tokens[i - 1]):
                sentence += ' '
            sentence += tok

        return sentence


def create_all_synonyms(synonyms: Dict, env=None, templates=None, use_features=None, goal_list=None) -> Dict: 
    synonyms_dict = defaultdict(set) 
    if env is not None:
        templates = env._instr_templates
        use_features = env._features.keys()
        if env._task_id == "unlock":
            goal_list = deepcopy(env._all_door_ftr_combs)
        else:
            goal_list = deepcopy(env._all_obj_ftr_combs)

    synonyms_list = set() 
    for goal in goal_list: 
        goal_tuple = (goal['colour'], goal['shape'])
        goal_synonyms = create_synonyms(goal=goal, templates=templates, use_features=use_features, synonyms=synonyms)
        synonyms_dict[goal_tuple] = synonyms_dict[goal_tuple].union(goal_synonyms) 
        synonyms_list = synonyms_list.union(goal_synonyms) 
    return synonyms_dict, list(synonyms_list)


def create_synonyms(goal: Union[Shape|Door], templates: List, synonyms: Dict) -> Dict: 
    goal_synonyms = set([])
    for template in templates:
        instr = generate_instruction(instr=deepcopy(template), goal=deepcopy(goal))
        for keyword, feature_synonyms in synonyms.items():
            if keyword in instr: 
                instr_2 = deepcopy(instr)
                for synonym in feature_synonyms: 
                    instr_3 = deepcopy(instr_2) 
                    final_instr = instr_3.replace(keyword, synonym)  
                    goal_synonyms.add(final_instr)
    return list(goal_synonyms)


class MultitaskShapes(gym.Env):
    def __init__(self, obj_ftr_combs: List, door_ftr_combs: List, grid:List[List], task_prog:List, task_templates:Dict, features:Dict,  goal_rsmpl_t:int, task_rsmpl_t:int, store_path:str, max_steps:int=None, slip_chance:float=0, seed:int=0):
        self.rng = np.random.default_rng(seed)
        self._num_objs = np.equal(np.array(grid), 'O').sum() + np.equal(np.array(grid), 'K').sum()
        self._num_doors = np.equal(np.array(grid), 'D').sum()

        # Key mask tells us whether to generate a key or another shape when generating
        # the list of all objects for a specific task. Keys are treated as objects in all regards,
        # but this mask is necessary to ensure keys aren't locked inside rooms they unlock
        self._key_mask = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 'O':
                    self._key_mask.append(False)
                elif grid[i][j] == 'K':
                    self._key_mask.append(True)

        # Task progression for compositional generalisation must be provided (not all make sense)
        self._tasks = task_prog
        # There is a specific task which teaches unlocking doors. Before it is learned, all doors spawn unlocked
        self._door_lock_status = {}
        unlock_task_found = False
        for task_id in self._tasks:
            if task_id == 'unlock':
                unlock_task_found = True
            self._door_lock_status[task_id] = unlock_task_found

        self._task_cnt = 0
        self._task_id = self._tasks[self._task_cnt]

        # We keep track of feature combinations the agent has already been trained on.
        # There are two cases: when the goal object is a door and when the goal object is something else
        self._all_obj_ftr_combs = deepcopy(obj_ftr_combs)
        self._all_door_ftr_combs = deepcopy(door_ftr_combs)
        if self._task_id == 'unlock':
            self._remaining_ftr_combs = deepcopy(door_ftr_combs)
        else:
            self._remaining_ftr_combs = deepcopy(obj_ftr_combs)
        self._sampled_ftr_combs = []

        self._features = features
        self._task_templates = task_templates
        
        objects, doors, self._instr = self._sample_task(reuse_goal=False, task_id=self._task_id)
        
        self._episode_num = 0
        self.goal_rsmpl_t = goal_rsmpl_t
        self._goal_rsmpl = False if self.goal_rsmpl_t is None else True
        self.task_rsmpl_t = task_rsmpl_t
        self._task_rsmpl = False if self.task_rsmpl_t is None else True

        self._grid = grid
        self._features = features
        self._store_path = store_path
        self._max_steps = max_steps
        self._slip_chance = slip_chance
        self._seed = seed

        if self._task_id == "go_to":
            constructor = ShapesGoto
        elif self._task_id == "pick_up":
            constructor = ShapesPickup
        elif self._task_id == 'unlock':
            constructor = ShapesUnlock
        elif self._task_id == "retrieve":
            constructor = ShapesRetrieve
        else:
            raise ValueError(f"Task id {self._task_id} not among the known ones.")
        
        self._env: Shapes = constructor(
                objects=objects,
                doors=doors,
                grid=self._grid,
                features=self._features,
                store_path=self._store_path,
                max_steps=self._max_steps,
                slip_chance=self._slip_chance,
                seed=self._seed
            )

        self._observation_space = spaces.Dict({
            "features": self._env.observation_space,
            "instr": spaces.Text(max_length=100)
            })
        
        self.done = False

    @property
    def instruction(self):
        return self._instr
    
    @property
    def task_id(self):
        return self._task_id

    @property
    def obs(self) -> np.ndarray:
        ret = {
            "features": self._env.obs,
            "instr": self._instr
        }
        return ret
        
    @property 
    def wall_mask(self) -> np.ndarray:
        return self._env.wall_mask
    
    @property
    def grid_shape(self) -> np.ndarray:
        return self._env.grid_shape
    
    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def agent_location(self) -> Tuple:
        return self._env.agent_location
    
    @property
    def goal(self):
        return self._env.goal

    @property
    def info(self):
        info = {
            "episode_num": self._episode_num,
            "task_id": self._task_id,
            "instruction": self._instr,
            "goal": str(self.goal),
            "tasks_exhausted": self.done
        }
        return info

    def _sample_doors(self, reuse_goal:bool, task_id:int):
        sampled_doors = []
        goal = None
        
        if task_id == 'unlock':
            if reuse_goal:
                # Doors are mutable, safer to create a new Shape
                goal = Door(colour=self.goal.colour, is_goal=True)
            else:
                idx = self.rng.integers(0, len(self._remaining_ftr_combs))
                goal_ftr_comb = self._remaining_ftr_combs.pop(idx)
                goal = Door(colour=goal_ftr_comb.colour, is_goal=True)
            sampled_doors.append(goal)
        else:
            # Goal will be sampled later, when objects are sampled
            pass

        while len(sampled_doors) < self._num_doors:
            idx = self.rng.integers(0, len(self._all_door_ftr_combs))
            door_comb = self._all_door_ftr_combs[idx]
            door = Door(colour=door_comb.colour, locked=self._door_lock_status[self._task_id])
            
            if door not in sampled_doors:
                sampled_doors.append(door)
            else:
                del door
        
        self.rng.shuffle(sampled_doors)
        return sampled_doors, goal

    def _sample_objects(self, reuse_goal:bool, task_id:int, doors: List[Door]):
        goal = None
        sampled_objs = []
        if task_id != 'unlock':
            if reuse_goal:
                # Shapes are mutable, safer to create a new Shape
                goal = Shape(shape=self.goal.shape, colour=self.goal.colour, is_goal=True)
            else:
                idx = self.rng.integers(0, len(self._remaining_ftr_combs))
                goal_ftr_comb = self._remaining_ftr_combs.pop(idx)
                goal = Shape(shape=goal_ftr_comb.shape, colour=goal_ftr_comb.colour, is_goal=True)
            sampled_objs.append(goal)
        else:
            # Door is the goal, it will have been sampled already
            pass

        # Important keys that unlock doors
        sampled_keys = []
        for door in doors:
            key = Shape(shape="key", colour=door.colour)
            # If the key that unlocks a door is itself a goal, put it into the corresponding list
            if goal is not None and key == goal:
                sampled_objs.remove(goal)
                sampled_keys.append(goal)
            else:
                sampled_keys.append(key)
        
        # Sampled objects can be keys (but that unlock no doors!)
        while len(sampled_objs) + len(sampled_keys) < self._num_objs:
            idx = self.rng.integers(0, len(self._all_obj_ftr_combs))
            obj_template = self._all_obj_ftr_combs[idx]
            obj = Shape(shape=obj_template.shape, colour=obj_template.colour)
            
            if obj not in sampled_objs and obj not in sampled_keys:
                sampled_objs.append(obj)
            else:
                del obj
        
        self.rng.shuffle(sampled_objs)
        self.rng.shuffle(sampled_keys)

        # The key must not spawn behind the door it unlocks. Since order of objects matters 
        # (they are placed in order by traversing the map from the top left corner). We must ensure that keys are reachable.
        # self._key_mask assures that by preserving which indices in the object list must be keys.
        assert(len(sampled_objs) + len(sampled_keys) == len(self._key_mask))
        ordered_objs = []
        for key_indicator in self._key_mask:
            obj: Shape
            if key_indicator:
                obj = sampled_keys.pop()
            else:
                obj = sampled_objs.pop()
            ordered_objs.append(obj)
            
        assert(len(sampled_objs) == 0)
        assert(len(sampled_keys) == 0)
        return ordered_objs, goal

    def _sample_task(self, reuse_goal:bool, task_id:int) -> List:
        doors, door_goal = self._sample_doors(reuse_goal=reuse_goal, task_id=task_id)
        objects, obj_goal = self._sample_objects(reuse_goal=reuse_goal, task_id=task_id, doors=doors)
        
        if task_id == 'unlock':
            goal = door_goal
        else:
            goal = obj_goal
        
        assert(goal is not None)

        instr = self.rng.choice(self._task_templates[self._task_id])

        instr = generate_instruction(instr=instr, goal=goal)
        return objects, doors, instr
    
    def reset(self, seed=None, options: Optional[dict]={}):
        self._episode_num += 1
        
        resample_goal = options.get("resample_goal", False)
        resample_goal = resample_goal or (self._goal_rsmpl and self._episode_num % self.goal_rsmpl_t == 0)

        resample_task = options.get("resample_task", False)
        resample_task = resample_task or (self._task_rsmpl and self._episode_num % self.task_rsmpl_t == 0)

        if resample_task:
            # We need to resample the goal because some tasks target doors and some shapes
            resample_goal = True
            self._task_cnt += 1
            if self._task_cnt < len(self._tasks):
                self._task_id = self._tasks[self._task_cnt]
                if self._task_id == 'unlock':
                    self._remaining_ftr_combs = deepcopy(self._all_door_ftr_combs)
                else:
                    self._remaining_ftr_combs = deepcopy(self._all_obj_ftr_combs)
                self._sampled_ftr_combs = []
            else:
                self.done = True
        
        if self._task_id == "go_to":
            constructor = ShapesGoto
        elif self._task_id == "pick_up":
            constructor = ShapesPickup
        elif self._task_id == 'unlock':
            constructor = ShapesUnlock
        elif self._task_id == "retrieve":
            constructor = ShapesRetrieve
        else:
            raise ValueError(f"Task id {self._task_id} not among the known ones.")
        
        try:
            objects, doors, self._instr = self._sample_task(reuse_goal=not resample_goal, task_id=self._task_id)
            self._env = constructor(
                    objects=objects,
                    doors=doors,
                    grid=self._grid,
                    features=self._features,
                    store_path=self._store_path,
                    max_steps=self._max_steps,
                    slip_chance=self._slip_chance,
                    seed=seed
                )
        except ValueError:
            self.done = True

        return self.obs, self.info
    
    def step(self, action):
        _, reward, is_terminal, truncated, info = self._env.step(action=action)
        superinfo = self.info
        superinfo["success"] = info["success"]
        return self.obs, reward, is_terminal, truncated, superinfo
        
    def render_frame(self) -> np.ndarray:
        return self._env.render_frame()

    def store_frame(self, plot_name:str='table') -> None:
        self._env.store_frame(plot_name=plot_name)


class ShapesMultitaskFactory(ABC):
    def __init__(self, hparams: Dict, store_path:str):
        self._hparams = hparams
        self._store_path = store_path

        obj_train_set, door_train_set, obj_holdout_set, door_holdout_set = self._train_holdout_split()
        
        self._obj_train_set = obj_train_set
        self._door_train_set = door_train_set
        self._obj_holdout_set = obj_holdout_set
        self._door_holdout_set = door_holdout_set

    
    @abstractmethod
    def _train_holdout_split(self) -> Tuple[List]:
        pass

    def get_all_instructions(self, set_id="ALL"):
        instructions = []
        
        if set_id == "ALL" or set_id=="TRAIN":
            obj_candidates = deepcopy(self._obj_train_set)
            if set_id == "ALL":
                obj_candidates.extend(deepcopy(self._obj_holdout_set))
        elif set_id == "HOLDOUT":
            obj_candidates = deepcopy(self._obj_holdout_set)

        if set_id == "ALL" or set_id=="TRAIN":
            door_candidates = deepcopy(self._door_train_set)
            if set_id == "ALL":
                door_candidates.extend(deepcopy(self._door_holdout_set))
        elif set_id == "HOLDOUT":
            door_candidates = deepcopy(self._door_holdout_set)

        for task_id in self._hparams["task_progression"]:
            if task_id == "unlock":
                candidates = door_candidates
            else:
                candidates = obj_candidates
            
            for candidate in candidates:
                for template in self._hparams["tasks"][task_id]:
                    instr = generate_instruction(
                        instr=template, 
                        goal=candidate
                    )
                    instructions.append(instr)
        return list(set(instructions))
    
    def get_env(self, set_id:str) -> MultitaskShapes:
        """Generates MultitaskShapes environments split into train and holdout environments.
        Args:
            set_id (str): In {'TRAIN', 'HOLDOUT', 'HARD_HOLDOUT'}. 
            Each contains disjoint sets of certain environment properties. 
            purpose (str): In {'TRAIN', 'EVAL'}.
        Returns:
            MultitaskShapes
        """
        if set_id == 'TRAIN':
            obj_ftr_combs = self._obj_train_set
            door_ftr_combs = self._door_train_set
        elif set_id == 'HOLDOUT':
            obj_ftr_combs = self._obj_holdout_set
            door_ftr_combs = self._door_holdout_set
        else:
            raise ValueError(f'set_id={set_id} not in [TRAIN, HOLDOUT, HARD_HOLDOUT].')
        
        goal_resample_t = self._hparams["goal_resample_t"]
        task_resample_t = self._hparams["task_resample_t"]
        
        env = MultitaskShapes(
                obj_ftr_combs=obj_ftr_combs,
                door_ftr_combs=door_ftr_combs,
                grid=self._hparams["grid"], 
                task_prog=self._hparams["task_progression"],
                task_templates=self._hparams["tasks"],
                goal_rsmpl_t=goal_resample_t,
                task_rsmpl_t=task_resample_t,
                features=self._hparams["features"],
                store_path=self._store_path, 
                max_steps=self._hparams["max_steps"], 
                slip_chance=self._hparams["slip_chance"], 
                seed=self._hparams["seed"]
            )
        return env 


# Test symbol grounding by reserving certain feature combinations
class ShapesAttrCombFactory(ShapesMultitaskFactory):
    def __init__(self, hparams, store_path):
        self._holdout_combs = hparams.get("reserved_combinations", [])
        if self._holdout_combs is None:
            self._holdout_combs = []
        super().__init__(hparams, store_path)
    
    def _train_holdout_split(self) -> Tuple[List]:
        
        obj_train_set = [] 
        door_train_set = [] 
        obj_holdout_set = [] 
        door_holdout_set = []

        all_combs = [dict(zip(self._hparams["features"].keys(), values)) for values in product(*self._hparams["features"].values())]
        
        for comb in all_combs:
            if comb["shape"] == "door":
                candidate = Door(colour=comb["colour"])
                if comb not in self._holdout_combs:
                    door_train_set.append(candidate)
                else:
                    door_holdout_set.append(candidate)
            else:
                candidate = Shape(colour=comb["colour"], shape=comb["shape"])
                if comb not in self._holdout_combs:
                    obj_train_set.append(candidate)
                else:
                    obj_holdout_set.append(candidate)
      
        return obj_train_set, door_train_set, obj_holdout_set, door_holdout_set

    
if __name__ == "__main__":
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, config_name="shapes_cmp")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    env_factory = ShapesAttrCombFactory(
        hparams=hparams, 
        store_path=store_path
    )

    env: MultitaskShapes = env_factory.get_env(set_id='TRAIN')
    instrs = env_factory.get_all_instructions()
    
    for episode in tqdm(range(10)):
        obs, _ = env.reset(options={"goal": DEFAULT_OBJECTS[0]})
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
        env.store_frame(plot_name=f"final_step_multitask_{env.task_id}")