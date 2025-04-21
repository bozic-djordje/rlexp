from typing import Dict, List, Tuple
import os
import numpy as np
import gymnasium as gym
import random
import re
from copy import deepcopy
from envs.taxicab.feature_taxicab import FeatureTaxicab, DEFAULT_FEATURES


class LanguageTaxicab(gym.Env):
    def __init__(self, hparams: Dict, store_path:str, synonyms: Dict, common_adjective_combs: List, holdout_adjective_combs: List=None):
        
        hparams["pomdp"] = True
        self._env = FeatureTaxicab(
            hparams=hparams,
            location_features=DEFAULT_FEATURES,
            store_path=store_path
        )

        self._common_adj_combs = common_adjective_combs
        self._hold_adj_combs = holdout_adjective_combs
        
        self._instr_synonyms = synonyms["instruction"]
        self._pass_synonyms = synonyms["passenger"]
        self._drive_synonyms = synonyms["drive"]
        self._location_synonyms = synonyms["location"]
        
        self._task_num = 0

        # These will be properly initialised after reset()
        self._task = None
        self._instr = None
        self._origin_ind = None 
        self._dest_ind = None
        _ = self.reset()

    @property
    def instruction(self):
        return self._instr
    
    @property
    def task_num(self):
        return self._task_num

    @property
    def obs(self) -> gym.spaces.MultiDiscrete:
        return self._env.obs, self._instr
        
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
        return self._env.observation_space
    
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
    
    def _sample_task(self) -> Tuple[Dict, str]:
        common_adj_combs = deepcopy(self._common_adj_combs)
        poi_1 = self._env.rng.choice(common_adj_combs, replace=False)

        if self._hold_adj_combs:
            hold_adj_combs = deepcopy(self._hold_adj_combs)
            poi_2 = self._env.rng.choice(hold_adj_combs, replace=False)
        else:
            poi_2 = self._env.rng.choice(common_adj_combs, replace=False)
        
        if self._env.rng.random() > 0.5:
            destination = poi_1
            origin = poi_2
        else:
            destination = poi_2
            origin = poi_1
        
        conf_1 = self._env.rng.choice(common_adj_combs, replace=False)
        conf_2 = self._env.rng.choice(common_adj_combs, replace=False)
        task = [origin, destination, conf_1, conf_2]
        
        self._env.rng.shuffle(task)
        origin_index = task.index(origin)
        dest_index = task.index(destination)
        
        instr = self._sample_instructions(adjective_combs=[origin, destination])
        return task, instr, origin_index, dest_index
    
    def reset(self, seed=None, options=None):
        self._task, self._instr, self._origin_ind, self._dest_ind = self._sample_task()
        options = {}
        options["location_features"] = self._task
        options["origin_ind"] = self._origin_ind
        options["dest_ind"] = self._dest_ind
        self._task_num += 1
        return self._env.reset(seed, options)
    
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
        # Attribute-value combinations that can be varied in the hard test set (others are fixed)
        self._h_var_attr_combs: Dict = deepcopy(self._nh_attr_combs)
        self._h_fix_attr_combs: Dict = hparams["hard_test_attributes"]
        self._train_test_split = hparams["train_test_split"]
        htest_task = {}
        
        for attr_name, attr_value in self._h_fix_attr_combs.items():
            self._nh_attr_combs[attr_name].remove(attr_value)
            self._h_var_attr_combs.pop(attr_name)
            htest_task[attr_name] = attr_value
        
        self._hard_holdout_adj_combs = self._get_hard_adjective_combinations(htest_template=htest_task)
        self._common_adj_combs, self._holdout_adj_combs = self._get_common_and_holdout_adjective_combinations()
        
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
    
    def _get_hard_adjective_combinations(self, htest_template: Dict) -> List:
        hhold_ftr_combos = []
        # This code expects exactly two attr names
        attr_names = list(self._h_var_attr_combs.keys())
        for attr_value_0 in self._h_var_attr_combs[attr_names[0]]:
            for attr_value_1 in self._h_var_attr_combs[attr_names[1]]:
                sample = deepcopy(htest_template)
                sample[attr_names[0]] = attr_value_0
                sample[attr_names[1]] = attr_value_1
                hhold_ftr_combos.append(sample)          
        return hhold_ftr_combos

    def _get_common_and_holdout_adjective_combinations(self) -> Tuple[List]:
        common_adj_combs = []
        holdout_adj_combs = []

        attr_names = list(self._nh_attr_combs.keys())
        for attr_value_0 in self._nh_attr_combs[attr_names[0]]:
            for attr_value_1 in self._nh_attr_combs[attr_names[1]]:
                for attr_value_2 in self._nh_attr_combs[attr_names[2]]:
                    for attr_value_3 in self._nh_attr_combs[attr_names[3]]:
                        ftr_comb = {}
                        ftr_comb[attr_names[0]] = attr_value_0
                        ftr_comb[attr_names[1]] = attr_value_1
                        ftr_comb[attr_names[2]] = attr_value_2
                        ftr_comb[attr_names[3]] = attr_value_3

                        set_id = random.random()
                        if set_id < self._train_test_split[0]:
                            common_adj_combs.append(ftr_comb)
                        else:
                            holdout_adj_combs.append(ftr_comb)
        return common_adj_combs, holdout_adj_combs
    
    def get_env(self, set_id:int) -> LanguageTaxicab:
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
                holdout_adjective_combs=None
            )
        elif set_id == 'HOLDOUT':
            env = LanguageTaxicab(
                hparams=self._hparams,
                store_path=self._store_path,
                synonyms=self._synonyms,
                common_adjective_combs=self._common_adj_combs,
                holdout_adjective_combs=self._holdout_adj_combs
            )
        elif set_id == 'HARD_HOLDOUT':
            env = LanguageTaxicab(
                hparams=self._hparams,
                store_path=self._store_path,
                synonyms=self._synonyms,
                common_adjective_combs=self._common_adj_combs,
                holdout_adjective_combs=self._hard_holdout_adj_combs
            )
        else:
            raise ValueError(f'set_id={set_id} not in [TRAIN, HOLDOUT, HARD_HOLDOUT].')
        
        return env

    
if __name__ == "__main__":
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, script_name="taxicab.py")
    
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

    env: LanguageTaxicab = env_factory.get_env(set_id='TRAIN')
    
    for episode in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
        env.store_frame(plot_name=f"final_step_multitask_{env.task_num}", use_png=True)