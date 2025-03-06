import os
from typing import Any, Dict, List, Tuple
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import random
import re


def substitute_placeholders(text, data):
    # Substitute passenger_reference, drive_common, and location_formulation
    text = re.sub(r"\bpassenger_reference\b", lambda _: random.choice(data["passenger_reference"]), text)
    text = re.sub(r"\bdrive_common\b", lambda _: random.choice(data["drive_common"]), text)
    text = re.sub(r"\blocation_formulation\b", lambda _: random.choice(data["location_formulation"]), text)

    # Substitute attributes like colour, novelty, direction, and building
    for key in ["colour", "novelty", "direction", "building", "size"]:
        text = re.sub(rf"\b{key}\b", lambda _: random.choice(data[key]), text)

    return text


def construct_hard_test(htest_sample: Dict, non_hard_test_attributes: Dict) -> List:
    samples = []
    # This code expect exactly two attr names
    attr_names = list(non_hard_test_attributes.keys())
    for attr_value_0 in non_hard_test_attributes[attr_names[0]]:
        for attr_value_1 in non_hard_test_attributes[attr_names[1]]:
            sample = deepcopy(htest_sample)
            sample[attr_names[0]] = attr_value_0
            sample[attr_names[1]] = attr_value_1
            samples.append(sample)          
    return samples


def construct_train_test(train_test_dict: Dict, split: List) -> Tuple[List]:
    train_samples = []
    test_samples = []

    attr_names = list(train_test_dict.keys())
    for attr_value_0 in train_test_dict[attr_names[0]]:
        for attr_value_1 in train_test_dict[attr_names[1]]:
            for attr_value_2 in train_test_dict[attr_names[2]]:
                for attr_value_3 in train_test_dict[attr_names[3]]:
                    sample = {}
                    sample[attr_names[0]] = attr_value_0
                    sample[attr_names[1]] = attr_value_1
                    sample[attr_names[2]] = attr_value_2
                    sample[attr_names[3]] = attr_value_3

                    set_id = random.random()
                    if set_id < split[0]:
                        train_samples.append(sample)
                    else:
                        test_samples.append(sample)
    return train_samples, test_samples
                    

if __name__ == "__main__":
    from utils import setup_artefact_paths
    from copy import deepcopy

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, script_name="taxicab.py")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    train_test_dict = {
        "colour": hparams["colour"],
        "building": hparams["building"],
        "size": hparams["size"],
        "fill": hparams["fill"]
    }
    non_hard_test_attributes = deepcopy(train_test_dict)
    htest_sample = {}
    hard_test_attributes = hparams["hard_test_attributes"]
    
    for attr_name, attr_value in hard_test_attributes.items():
        train_test_dict[attr_name].remove(attr_value)
        non_hard_test_attributes.pop(attr_name)
        htest_sample[attr_name] = attr_value
    
    hard_test_samples = construct_hard_test(
        htest_sample=htest_sample, 
        non_hard_test_attributes=non_hard_test_attributes
    )

    split = [0.7, 0.3]
    train_samples, test_samples = construct_train_test(
        train_test_dict=train_test_dict, 
        split=split
    )
    print(0)