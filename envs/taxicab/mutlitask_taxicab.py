from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import gymnasium as gym
import random
import re
from copy import deepcopy
from envs.taxicab.single_taxicab import FeatureTaxicab


class MultitaskTaxicab(gym.Env):
    def __init__(self, hparams: Dict, store_path:str):
        # Attribute-value combinations allowed in train and (non-hard) test set
        self.nh_attr_combs = deepcopy({name: hparams[name] for name in hparams["attribute_order"]})
        # Attribute-value combinations that can be varied in the hard test set (others are fixed)
        self.h_var_attr_combs: Dict = deepcopy(self.nh_attr_combs)
        self.h_fix_attr_combs: Dict = hparams["hard_test_attributes"]
        self.train_test_split = hparams["train_test_split"]
        htest_task = {}
        
        for attr_name, attr_value in self.h_fix_attr_combs.items():
            self.nh_attr_combs[attr_name].remove(attr_value)
            self.h_var_attr_combs.pop(attr_name)
            htest_task[attr_name] = attr_value
        
        self.hholdout_ftr_combs = self._construct_hard_featre_combs(htest_template=htest_task)
        self.common_ftr_combs, self.holdout_ftr_combs = self._construct_common_and_holdout_feature_combs()
        
        # Natural language options. At the beginning of each episode the exact instruction
        # will be constructed by sampling from these options.
        self.pass_synonyms = hparams["passenger_formulation"]
        self.drive_synonyms = hparams["drive_formulation"]
        self.location_synonyms = hparams["location_formulation"]
        self.instr_synonyms = hparams["goal_formulation"]
        
        # Seeding random generators for reproducibility
        if hparams["seed"] is not None:
            self.rng = np.random.default_rng(hparams["seed"])
        else:
            self.rng = np.random.default_rng()
        
        self.task_num = 0
        self.cur_task, self.cur_instr, origin_index, dest_index = self.sample_task(set_id="train")
        self.env = FeatureTaxicab(
            hparams=hparams,
            location_features=self.cur_task,
            origin_ind=origin_index,
            dest_ind=dest_index,
            store_path=store_path
        )

    @property
    def obs(self) -> gym.spaces.MultiDiscrete:
        return self.env.obs

    @property
    def wall_mask(self) -> np.ndarray:
        return self.env.wall_mask

    @property
    def grid_shape(self) -> Tuple:
        return self.env.grid_shape
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    def _construct_hard_featre_combs(self, htest_template: Dict) -> List:
        hhold_ftr_combos = []
        # This code expect exactly two attr names
        attr_names = list(self.h_var_attr_combs.keys())
        for attr_value_0 in self.h_var_attr_combs[attr_names[0]]:
            for attr_value_1 in self.h_var_attr_combs[attr_names[1]]:
                sample = deepcopy(htest_template)
                sample[attr_names[0]] = attr_value_0
                sample[attr_names[1]] = attr_value_1
                hhold_ftr_combos.append(sample)          
        return hhold_ftr_combos

    def _construct_common_and_holdout_feature_combs(self) -> Tuple[List]:
        common_ftr_combs = []
        holdout_ftr_combs = []

        attr_names = list(self.nh_attr_combs.keys())
        for attr_value_0 in self.nh_attr_combs[attr_names[0]]:
            for attr_value_1 in self.nh_attr_combs[attr_names[1]]:
                for attr_value_2 in self.nh_attr_combs[attr_names[2]]:
                    for attr_value_3 in self.nh_attr_combs[attr_names[3]]:
                        ftr_comb = {}
                        ftr_comb[attr_names[0]] = attr_value_0
                        ftr_comb[attr_names[1]] = attr_value_1
                        ftr_comb[attr_names[2]] = attr_value_2
                        ftr_comb[attr_names[3]] = attr_value_3

                        set_id = random.random()
                        if set_id < self.train_test_split[0]:
                            common_ftr_combs.append(ftr_comb)
                        else:
                            holdout_ftr_combs.append(ftr_comb)
        return common_ftr_combs, holdout_ftr_combs
    
    def _sample_instructions(self, task_attr_combs:List):
        instr = random.choice(self.instr_synonyms)
        text = instr["phrase"]
        goal_first = instr["goal_first"]

        # Substitute passenger_reference, drive_common, and location_formulation
        text = re.sub(r"\bpassenger_reference\b", lambda _: random.choice(self.pass_synonyms), text)
        text = re.sub(r"\bdrive_common\b", lambda _: random.choice(self.drive_synonyms), text)
        text = re.sub(r"\blocation_formulation\b", lambda _: random.choice(self.location_synonyms), text)

        if goal_first:
            task_attr_combs.reverse()
        
        for attr_comb in task_attr_combs:
            for key in attr_comb.keys():
                # This skips attribute values which are non-strings. 
                # Their keys will never exist in text, but re will throw an error
                if isinstance(attr_comb[key], str):
                    text = re.sub(rf"\b{key}\b", attr_comb[key], text, count=1) 
        return text
    
    def sample_task(self, set_id:str) -> Tuple[Dict, str]:
        common_feature_combs = deepcopy(self.common_ftr_combs)
        if set_id == "common":
            poi_1 = np.random.choice(common_feature_combs, replace=False)
            poi_2 = np.random.choice(common_feature_combs, replace=False)
        # Only one of the points of interest will come from the unobserved set
        elif set_id == "holdout":
            poi_1 = np.random.choice(self.holdout_ftr_combs)
            poi_2 = np.random.choice(common_feature_combs, replace=False)
        else:
            poi_1 = np.random.choice(self.hholdout_ftr_combs)
            poi_2 = np.random.choice(common_feature_combs, replace=False)
        
        if np.random.rand() > 0.5:
            destination = poi_1
            origin = poi_2
        else:
            destination = poi_2
            origin = poi_1
        
        conf_1 = np.random.choice(common_feature_combs, replace=False)
        conf_2 = np.random.choice(common_feature_combs, replace=False)
        task = [origin, destination, conf_1, conf_2]
        
        random.shuffle(task)
        origin_index = task.index(origin)
        destination_index = task.index(destination)
        
        instr = self._sample_instructions(task_attr_combs=[origin, destination])
        return task, instr, origin_index, destination_index
    
    def reset(self, seed: Optional[int]=None, options:Optional[dict]=None):
        if options and "set_id" in options:
            self.cur_task, self.cur_instr, origin_index, dest_index = self.sample_task(set_id=options["set_id"])
            options["location_features"] = self.cur_task
            options["origin_ind"] = origin_index
            options["dest_ind"] = dest_index
        else:
            options = None
        self.task_num += 1
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        return self.env.step(action=action)
    
    def render_frame(self, use_png:bool=False) -> np.ndarray:
        return self.env.render_frame(use_png=use_png)

    def store_frame(self, plot_name:str='table', use_png:bool=True) -> None:
        self.env.store_frame(plot_name=plot_name, use_png=use_png)
    
if __name__ == "__main__":
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, script_name="taxicab.py")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    env = MultitaskTaxicab(
        hparams=hparams, 
        store_path=store_path
    )
    options = {"set_id": "train"}
    for episode in tqdm(range(hparams['n_episodes'])):
        # TODO: It doesn't resample goals properly!
        obs, _ = env.reset(options=options)
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
        env.store_frame(plot_name=f"final_step_multitask_{env.task_num}", use_png=True)