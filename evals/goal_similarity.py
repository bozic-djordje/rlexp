from collections import defaultdict
from copy import deepcopy
import os
from envs.shapes.multitask_shapes import MultitaskShapes, generate_instruction, ShapesAttrCombFactory

    
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
    possible_goals = env.goal_list
    
    synonyms_list = []
    synonyms_dict = defaultdict(list)
    for goal in possible_goals:
        goal_tuple = (goal['colour'], goal['shape'])
        for template in env._instr_templates:
            instr = generate_instruction(instr=deepcopy(template), goal=deepcopy(goal), all_feature_keys=env._features.keys())
            for keyword, synonyms in hparams["synonyms"].items():
                if keyword in instr:
                    instr_2 = deepcopy(instr)
                    for synonym in synonyms:
                        instr_3 = deepcopy(instr_2)
                        final_instr = instr_3.replace(keyword, synonym)
                        synonyms_list.append(final_instr)
                        synonyms_dict[goal_tuple].append(final_instr)
    print(len(synonyms_list))
