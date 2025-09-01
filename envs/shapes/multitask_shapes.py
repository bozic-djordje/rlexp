from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Set, Tuple
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import re
from copy import deepcopy
from envs.shapes.shapes import ShapesGoto, ShapesGotoEasy, ShapesPickup, ShapesRetrieve, DEFAULT_OBJECTS


def generate_instruction(instr: str, goal: dict, all_feature_keys: list) -> str:
        # Split the sentence into words and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', instr)

        result_tokens = []
        for token in tokens:
            if token in goal:
                result_tokens.append(goal[token])
            elif token in all_feature_keys:
                # It's a placeholder, but not provided — skip it
                continue
            else:
                result_tokens.append(token)

        # Reconstruct sentence with spacing
        sentence = ''
        for i, tok in enumerate(result_tokens):
            if i > 0 and re.match(r'\w', tok) and re.match(r'\w', result_tokens[i - 1]):
                sentence += ' '
            sentence += tok

        return sentence


def create_all_synonyms(env, synonyms: Dict) -> Dict: 
    synonyms_dict = defaultdict(set) 
    synonyms_list = set() 
    for goal in env.goal_list: 
        goal_tuple = (goal['colour'], goal['shape'])
        goal_synonyms = create_synonyms(goal=goal, templates=env._instr_templates, use_features=env._features.keys(), synonyms=synonyms)
        synonyms_dict[goal_tuple] = synonyms_dict[goal_tuple].union(goal_synonyms) 
        synonyms_list = synonyms_list.union(goal_synonyms) 
    return synonyms_dict, list(synonyms_list)


def create_synonyms(goal: Dict, templates: List, synonyms: Dict, use_features: Set) -> Dict: 
    goal_synonyms = set([])
    for template in templates:
        instr = generate_instruction(instr=deepcopy(template), goal=deepcopy(goal), all_feature_keys=use_features)
        for keyword, feature_synonyms in synonyms.items():
            if keyword in instr: 
                instr_2 = deepcopy(instr)
                for synonym in feature_synonyms: 
                    instr_3 = deepcopy(instr_2) 
                    final_instr = instr_3.replace(keyword, synonym)  
                    goal_synonyms.add(final_instr)
    return list(goal_synonyms)


class MultitaskShapes(gym.Env):
    def __init__(self, allowed_objects: List, grid:List[List], task_id:str, instruction_templates:List, feature_order:List, features:Dict, num_objects: int, resample_interval:int, store_path:str, default_feature:int=0, max_steps:int=200, slip_chance:float=0, goal_channel:bool=False, change_loc_on_fix_goal:bool=False, obs_type:str="box", seed:int=0):
        self.rng = np.random.default_rng(seed)
        
        self._num_objects = num_objects
        self._allowed_objects = allowed_objects

        self._feature_order = feature_order
        self._features = features

        self._instr_templates = instruction_templates
        
        self._objects, self._instr = self._sample_task()
        self._goal_obj = self._objects[0]
        self._change_loc = change_loc_on_fix_goal

        self._task_num = 0
        self._resample_interval = resample_interval

        if task_id == "go_to":
            self._env = ShapesGoto(
                objects=self._objects,
                grid=grid,
                feature_order=feature_order,
                features=features,
                store_path=store_path,
                max_steps=max_steps,
                default_feature=default_feature,
                slip_chance=slip_chance,
                seed=seed,
                goal_channel=goal_channel,
                obs_type=obs_type
            )
        elif task_id == "go_to_easy":
            self._env = ShapesGotoEasy(
                objects=self._objects,
                grid=grid,
                feature_order=feature_order,
                features=features,
                store_path=store_path,
                max_steps=max_steps,
                default_feature=default_feature,
                slip_chance=slip_chance,
                seed=seed,
                goal_channel=goal_channel,
                obs_type=obs_type
            )
        elif task_id == "pick_up":
            pass
        elif task_id == "retrieve":
            pass
        else:
            raise ValueError(f"Task id {task_id} not among the known ones.")

        self._observation_space = spaces.Dict({
            "features": self._env.observation_space,
            "instr": spaces.Text(max_length=100)
            })

    @property
    def instruction(self):
        return self._instr
    
    @property
    def task_num(self):
        return self._task_num

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
    def goal_list(self) -> List:
        return deepcopy(self._allowed_objects)
    
    def set_resample_interval(self, interval:int) -> None:
        self._resample_interval = interval

    def _sample_objects(self, candidates, n, loc_key='loc', goal:Dict=None):
        seen_locs = set()
        seen_others = set()
        sampled = []

        # Use the same shape as a goal, but resample its location
        if goal is not None:
            if loc_key not in goal:
                candidates_cpy = deepcopy(candidates)
                goal_candidates = []
                
                cand: Dict
                for cand in candidates_cpy:
                    # Strip each candidate of irrelevant keys
                    loc = cand.pop(loc_key)
                    
                    if cand == goal:
                        cand[loc_key] = loc
                        goal_candidates.append(cand)
                
                self.rng.shuffle(goal_candidates)
                goal = goal_candidates[0]
            
            sampled.append(goal)
            seen_locs.add(goal[loc_key])
            others = tuple(sorted((k, v) for k, v in goal.items() if k != loc_key and k != "is_goal"))
            
            seen_others.add(others)

        indices = list(range(len(candidates)))
        self.rng.shuffle(indices)

        if len(sampled) == n:
            return sampled

        for idx in indices:
            d = candidates[idx]
            loc = d[loc_key]
            others = tuple(sorted((k, v) for k, v in d.items() if k != loc_key and k != "is_goal"))

            if loc in seen_locs or others in seen_others:
                continue

            sampled.append(d)
            seen_locs.add(loc)
            seen_others.add(others)

            if len(sampled) == n:
                break
        
        for sample in sampled:
            if "is_goal" in sample:
                d.pop("is_goal")

        return deepcopy(sampled)

    def _sample_task(self, goal:Dict=None) -> List:
        if goal is not None and "is_goal" in goal:
            goal.pop("is_goal")
        
        objects = self._sample_objects(candidates=self.goal_list, n=self._num_objects, goal=goal)
        instr = self.rng.choice(self._instr_templates)
        instr = generate_instruction(instr=instr, goal=objects[0], all_feature_keys=self._features.keys())
        
        objects[0]["is_goal"] = True
        self.rng.shuffle(objects)

        return objects, instr
    
    def reset(self, seed=None, options: Optional[dict]={}):
        self._task_num += 1
        if "goal" in options:
            goal = deepcopy(options["goal"])
            self._objects, self._instr = self._sample_task(goal=goal)
        else:
            if self._task_num % self._resample_interval == 0:
                self._objects, self._instr = self._sample_task()
            else:
                goal = deepcopy(self._objects[0])
                if self._change_loc:
                    goal.pop("loc")
                self._objects, self._instr = self._sample_task(goal=goal)
        
        _, info = self._env.reset(seed, options={"objects": self._objects})
        return self.obs, info
    
    def step(self, action):
        _, reward, is_terminal, truncated, info = self._env.step(action=action)
        return self.obs, reward, is_terminal, truncated, info
        
    def render_frame(self) -> np.ndarray:
        return self._env.render_frame()

    def store_frame(self, plot_name:str='table') -> None:
        self._env.store_frame(plot_name=plot_name)


# TODO: Remove this comment. Shapes multitask factory splitting train/test sets based on *some* criteria. See your notebook.
class ShapesMultitaskFactory(ABC):
    def __init__(self, hparams: Dict, store_path:str):
        self._hparams = deepcopy(hparams)
        self._store_path = store_path
        
        self._train_set, self._holdout_set = self._train_holdout_split(grid=hparams["grid"])
    
    @abstractmethod
    def _train_holdout_split(self, grid: List[List]) -> Tuple[List]:
        pass

    def get_all_instructions(self):
        instructions = []
        candidates = deepcopy(self._train_set)
        candidates.extend(self._holdout_set)
        
        for candidate in candidates:
            for template in self._hparams[self._hparams["task_id"]]:
                instr = generate_instruction(
                    instr=template, 
                    goal=candidate, 
                    all_feature_keys=self._hparams["features"].keys()
                )
                instructions.append(instr)
        return list(set(instructions))
    
    def get_env(self, set_id:str, purpose:str='TRAIN') -> MultitaskShapes:
        """Generates MultitaskShapes environments split into train and holdout environments.
        Args:
            set_id (str): In {'TRAIN', 'HOLDOUT', 'HARD_HOLDOUT'}. 
            Each contains disjoint sets of certain environment properties. 
            purpose (str): In {'TRAIN', 'EVAL'}.
        Returns:
            MultitaskShapes
        """
        if set_id == 'TRAIN':
            allowed_objects = self._train_set
        elif set_id == 'HOLDOUT':
            allowed_objects = self._holdout_set
        else:
            raise ValueError(f'set_id={set_id} not in [TRAIN, HOLDOUT, HARD_HOLDOUT].')
        
        # If we are using the environment for evaluation, we want to resample tasks on every episode
        if purpose == 'EVAL':
            resample_interval = 1
        else:
            resample_interval = self._hparams["resample_episodes"]
        
        env = MultitaskShapes(
                allowed_objects=allowed_objects,
                grid=self._hparams["grid"], 
                task_id=self._hparams["task_id"], 
                instruction_templates=self._hparams[self._hparams["task_id"]],
                feature_order=self._hparams["use_features"], 
                features=self._hparams["features"],
                num_objects=self._hparams["num_objects"], 
                resample_interval= resample_interval, 
                store_path=self._store_path, 
                default_feature=self._hparams["default_feature"], 
                max_steps=self._hparams["max_steps"], 
                slip_chance=self._hparams["slip_chance"], 
                goal_channel=self._hparams["goal_channel"], 
                obs_type=self._hparams["obs_type"],
                seed=self._hparams["seed"]
            )
        return env 


# Test symbol grounding by reserving certain locations on the maps where goals can be spawned
class ShapesPositionFactory(ShapesMultitaskFactory):

    def _train_holdout_split(self, grid: List[List]) -> Tuple[List]:
        grid = np.array(grid)
        
        train_positions = np.where(grid == ' ')
        train_positions = list(zip(*train_positions))

        holdout_positions = np.where(grid == 'T')
        holdout_positions = list(zip(*holdout_positions))

        use_features = set(self._hparams["use_features"])
        train_features = deepcopy(self._hparams["features"])
        holdout_features = deepcopy(self._hparams["features"])
        keys_to_remove = [k for k in train_features if k not in use_features]
        
        for feature in keys_to_remove:
            train_features.pop(feature)
            holdout_features.pop(feature)
        
        train_features["loc"] = train_positions
        holdout_features["loc"] = holdout_positions

        train_candidates = [dict(zip(train_features.keys(), values)) for values in product(*train_features.values())]
        holdout_candidates = [dict(zip(holdout_features.keys(), values)) for values in product(*holdout_features.values())]

        return train_candidates, holdout_candidates


# Test symbol grounding by reserving certain feature combinations
class ShapesAttrCombFactory(ShapesMultitaskFactory):
    def __init__(self, hparams, store_path):
        self._holdout_combs = hparams["reserved_combinations"]
        super().__init__(hparams, store_path)
    
    def _train_holdout_split(self, grid: List[List]) -> Tuple[List]:
        grid = np.array(grid)
        
        positions = np.where(grid == 'T')
        positions = list(zip(positions[0], positions[1]))
        if len(positions) == 0:
            positions = np.where(grid == ' ')
            positions = list(zip(positions[0], positions[1]))

        use_features = set(self._hparams["use_features"])
        train_features: Dict = deepcopy(self._hparams["features"])
        keys_to_remove = [k for k in train_features if k not in use_features]
        
        for feature in keys_to_remove:
            train_features.pop(feature)
        
        train_features["loc"] = positions
        train_all = [dict(zip(train_features.keys(), values)) for values in product(*train_features.values())]
        n_candidates = len(train_all)
        
        holdout_candidates = []
        train_candidates = []
        for candiadte in train_all:
            c = deepcopy(candiadte)
            c.pop("loc")
            if c in self._holdout_combs:
                holdout_candidates.append(candiadte)
            else:
                train_candidates.append(candiadte)
        
        assert(n_candidates == len(train_candidates) + len(holdout_candidates))
        return train_candidates, holdout_candidates

    
if __name__ == "__main__":
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, config_name="shapes")
    
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
        env.store_frame(plot_name=f"final_step_multitask_{env.task_num}")