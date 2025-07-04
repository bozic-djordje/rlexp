from typing import Dict, List, Tuple
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import re
from copy import deepcopy
from envs.taxicab.feature_taxicab import FeatureTaxicab, DEFAULT_FEATURES


class LanguageTaxicab(gym.Env):
    def __init__(self, hparams: Dict, store_path:str, synonyms: Dict, common_adjective_combs: List, holdout_adjective_combs: List=None, force_pomdp:bool=True):
        
        if force_pomdp:
            hparams["pomdp"] = True
        
        self._easy_mode = hparams["easy_mode"]
        self._env = FeatureTaxicab(
            hparams=hparams,
            location_features=DEFAULT_FEATURES,
            store_path=store_path,
            easy_mode=hparams["easy_mode"]
        )

        self._common_adj_combs = common_adjective_combs
        self._hold_adj_combs = holdout_adjective_combs
        
        self._instr_synonyms = synonyms["instruction"]
        self._pass_synonyms = synonyms["passenger"]
        self._drive_synonyms = synonyms["drive"]
        self._location_synonyms = synonyms["location"]
        
        self._task_num = 0
        self._observation_space = spaces.Dict({
            "features": spaces.MultiDiscrete([10] * 9),
            "instr": spaces.Text(max_length=100)
            })

        # These will be properly initialised after reset()
        self._task = None
        self._instr = None
        self._origin_ind = None 
        self._dest_ind = None
        self._options = None
        _ = self.reset()

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
    
    def _sample_instructions(self, adjective_combs:List):
        instr = self._env.rng.choice(self._instr_synonyms)
        text = instr["phrase"]
        goal_first = instr["goal_first"]

        # Substitute passenger_reference, drive_common, and location_formulation
        text = re.sub(r"\bpassenger_reference\b", lambda _: random.choice(self._pass_synonyms), text)
        text = re.sub(r"\bdrive_common\b", lambda _: random.choice(self._drive_synonyms), text)
        text = re.sub(r"\blocation_formulation\b", lambda _: random.choice(self._location_synonyms), text)

        if goal_first:
            adjective_combs.reverse()
        
        for attr_comb in adjective_combs:
            for key in attr_comb.keys():
                # This skips attribute values which are non-strings. 
                # Their keys will never exist in text, but re will throw an error
                if isinstance(attr_comb[key], str):
                    text = re.sub(rf"\b{key}\b", attr_comb[key], text, count=1)
        return text

    def _sample_adj_comb(self, adj_combs: List) -> Tuple:
        poi = self._env.rng.choice(adj_combs, replace=False)
        adj_combs.remove(poi)

        # In easy mode each feature must not be present more than once on a map
        # This removes all addj combs with features already present in poi
        if self._easy_mode:
            adj_combs = [addj_comb for addj_comb in adj_combs 
                                if not any(addj_comb[k] == poi[k] for k in addj_comb.keys())]
        return poi, adj_combs

    # TODO: This creates the issue because origin == destination
    def _sample_task(self) -> Tuple[Dict, str]:
        common_adj_combs = deepcopy(self._common_adj_combs)
        
        if self._hold_adj_combs:
            hold_adj_combs = deepcopy(self._hold_adj_combs)
            poi_1 = self._env.rng.choice(hold_adj_combs, replace=False)
            if self._easy_mode:
                common_adj_combs = [addj_comb for addj_comb in common_adj_combs 
                                if not any(addj_comb[k] == poi_1[k] for k in addj_comb.keys())]
        else:
            poi_1, common_adj_combs = self._sample_adj_comb(adj_combs=common_adj_combs)

        poi_2, common_adj_combs = self._sample_adj_comb(adj_combs=common_adj_combs)
        if self._env.rng.random() > 0.5:
            destination = poi_1
            origin = poi_2
        else:
            destination = poi_2
            origin = poi_1
        
        assert(origin != destination)
        conf_1, common_adj_combs = self._sample_adj_comb(adj_combs=common_adj_combs)
        conf_2, common_adj_combs = self._sample_adj_comb(adj_combs=common_adj_combs)
        task = [origin, destination, conf_1, conf_2]
        
        self._env.rng.shuffle(task)
        origin_index = task.index(origin)
        dest_index = task.index(destination)
        
        instr = self._sample_instructions(adjective_combs=[origin, destination])
        return task, instr, origin_index, dest_index
    
    def reset(self, seed=None, options=None):
        # In easy mode only one configuration exists
        if self._options is not None and self._easy_mode:
            options = self._options
        else:
            self._task, self._instr, self._origin_ind, self._dest_ind = self._sample_task()
            options = {}
            options["location_features"] = self._task
            options["origin_ind"] = self._origin_ind
            options["dest_ind"] = self._dest_ind
            self._options = options
        self._task_num += 1
        _, info = self._env.reset(seed, options)
        # Hack to include instruction into obs
        return self.obs, info
    
    def step(self, action):
        _, reward, is_terminal, truncated, info = self._env.step(action=action)
        return self.obs, reward, is_terminal, truncated, info
        
    def render_frame(self, use_png:bool=True) -> np.ndarray:
        return self._env.render_frame(use_png=use_png)

    def store_frame(self, plot_name:str='table', use_png:bool=True) -> None:
        self._env.store_frame(plot_name=plot_name, use_png=use_png)

    
class LanguageTaxicabFactory:
    def __init__(self, hparams: Dict, store_path:str):
        self._hparams = deepcopy(hparams)
        self._store_path = store_path

        # Attribute-value combinations allowed in train and (non-hard) test set
        self._nh_attr_combs = deepcopy({name: hparams[name] for name in hparams["attribute_order"]})
        all_adj_combs = self._get_all_adj_combinations()
        
        self._common_adj_combs, self._holdout_adj_combs, self._hard_holdout_adj_combs = self.split_datasets(
            all_adj_combinations=all_adj_combs, 
            split=hparams["train_test_split"], 
            hard_attrs=hparams["hard_test_attributes"]
        )
        
        if "easy_mode" in hparams and hparams["easy_mode"]:
            self._common_adj_combs.extend(self._holdout_adj_combs)
            self._common_adj_combs.extend(self._hard_holdout_adj_combs)
        
        # Natural language options. At the beginning of each episode the exact instruction
        # will be constructed by sampling from these options.
        self._synonyms = {
            "passenger": hparams["passenger_formulation"],
            "drive": hparams["drive_formulation"],
            "location": hparams["location_formulation"],
            "instruction": hparams["goal_formulation"]
        }
    
    @property
    def common_adjective_combinations(self) -> List:
        return self._common_adj_combs
    
    @property
    def holdout_adjective_combinations(self) -> List:
        return self._holdout_adj_combs
    
    @property
    def hard_holdout_adjective_combinations(self) -> List:
        return self._hard_holdout_adj_combs

    def _get_all_adj_combinations(self) -> Tuple[List]:
        all_adj_combs = []

        attr_names = list(self._nh_attr_combs.keys())
        for attr_value_0 in self._nh_attr_combs[attr_names[0]]:
            for attr_value_1 in self._nh_attr_combs[attr_names[1]]:
                ftr_comb = {}
                ftr_comb[attr_names[0]] = attr_value_0
                ftr_comb[attr_names[1]] = attr_value_1
                all_adj_combs.append(ftr_comb)
        return all_adj_combs
    
    def split_datasets(self, all_adj_combinations: List, split: int, hard_attrs: Dict) -> Tuple:
        train_combs = []
        holdout_combs = []
        hholdout_combs = []

        for attr_comb in all_adj_combinations:
            if all(attr_comb[k] == hard_attrs[k] for k in hard_attrs.keys()):
                hholdout_combs.append(attr_comb)
            else:
                set_id = random.random()
                if set_id < split:
                    train_combs.append(attr_comb)
                else:
                    holdout_combs.append(attr_comb)
        return train_combs, holdout_combs, hholdout_combs
        
    def get_all_instructions(self):
        
        all_adjective_combs = deepcopy(self._common_adj_combs)
        all_adjective_combs.extend(deepcopy(self._holdout_adj_combs))
        all_adjective_combs.extend(deepcopy(self._hard_holdout_adj_combs))
        
        partials_1 = []
        for instr in self._synonyms["instruction"]:
            text = deepcopy(instr["phrase"])
            partials_1.append(text)
        
        partials_2 = []
        for p1 in partials_1:
            for passenger in self._synonyms["passenger"]:
                text = deepcopy(p1)
                text = re.sub(r"\bpassenger_reference\b", passenger, text)
                partials_2.append(text)
        
        partials_3 = []
        for p2 in partials_2:
            drive_count = len(re.findall(r"\bdrive_common\b", p2))
            if drive_count == 1:
                for drive in self._synonyms["drive"]:
                    text = deepcopy(p2)
                    text = re.sub(r"\bdrive_common\b", drive, text)
                    partials_3.append(text)
            else:
                for drive_1 in self._synonyms["drive"]:
                    for drive_2 in self._synonyms["drive"]:
                        text = deepcopy(p2)
                        text = re.sub(r"\bdrive_common\b", drive_1, text, count=1)
                        text = re.sub(r"\bdrive_common\b", drive_2, text, count=1)
                        partials_3.append(text)
        
        partials_4 = []
        for p3 in partials_3:
            for location_1 in self._synonyms["location"]:
                for location_2 in self._synonyms["location"]:
                    text = deepcopy(p3)
                    text = re.sub(r"\blocation_formulation\b", location_1, text, count=1)
                    text = re.sub(r"\blocation_formulation\b", location_2, text, count=1)
                    partials_4.append(text)

        partials_5 = []
        for p4 in partials_4:
            for place in all_adjective_combs:
                text = deepcopy(p4)
                for key in place.keys():
                    # This skips attribute values which are non-strings. 
                    # Their keys will never exist in text, but re will throw an error
                    if isinstance(place[key], str):
                        text = re.sub(rf"\b{key}\b", place[key], text, count=1)
                partials_5.append(text)
        
        partials_6 = []
        for p5 in partials_5:
            for place in all_adjective_combs:
                text = deepcopy(p5)
                for key in place.keys():
                    # This skips attribute values which are non-strings. 
                    # Their keys will never exist in text, but re will throw an error
                    if isinstance(place[key], str):
                        text = re.sub(rf"\b{key}\b", place[key], text, count=1)
                partials_6.append(text)

        instructions = set(partials_6)

        return list(instructions)

    def get_env(self, set_id:int, force_pomdp:bool=True) -> LanguageTaxicab:
        """Generates LanguageTaxicab environment with possible adjective combinations defined 
        by set_id.
        Args:
            set_id (int): In {'TRAIN', 'HOLDOUT', 'HARD_HOLDOUT'}. 
            Each contains disjoint sets adjective groupings. 

        Returns:
            LanguageTaxicab: LanguageTaxicab environment which can generate 
            instructions that uses the selected set of adjective groupings.
        """
        
        if set_id == 'TRAIN':
            env = LanguageTaxicab(
                hparams=self._hparams,
                store_path=self._store_path,
                synonyms=self._synonyms,
                common_adjective_combs=self._common_adj_combs,
                holdout_adjective_combs=None,
                force_pomdp=force_pomdp
            )
        elif set_id == 'HOLDOUT':
            env = LanguageTaxicab(
                hparams=self._hparams,
                store_path=self._store_path,
                synonyms=self._synonyms,
                common_adjective_combs=self._common_adj_combs,
                holdout_adjective_combs=self._holdout_adj_combs,
                force_pomdp=force_pomdp
            )
        elif set_id == 'HARD_HOLDOUT':
            env = LanguageTaxicab(
                hparams=self._hparams,
                store_path=self._store_path,
                synonyms=self._synonyms,
                common_adjective_combs=self._common_adj_combs,
                holdout_adjective_combs=self._hard_holdout_adj_combs,
                force_pomdp=force_pomdp
            )
        else:
            raise ValueError(f'set_id={set_id} not in [TRAIN, HOLDOUT, HARD_HOLDOUT].')
        
        return env

    
if __name__ == "__main__":
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, config_name="taxicab_easy")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    env_factory = LanguageTaxicabFactory(
        hparams=hparams, 
        store_path=store_path
    )
    
    print('Common adjective combinations are: ')
    print(env_factory.common_adjective_combinations)
    print()

    print('Holdout adjective combinations are: ')
    print(env_factory.holdout_adjective_combinations)
    print()

    print('Hard holdout adjective combinations are: ')
    print(env_factory.hard_holdout_adjective_combinations)
    print()

    possible_instrs = env_factory.get_all_instructions()

    env: LanguageTaxicab = env_factory.get_env(set_id='TRAIN')
    
    for episode in tqdm(range(10)):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
        env.store_frame(plot_name=f"final_step_multitask_{env.task_num}", use_png=True)