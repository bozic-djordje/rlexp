from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import re
from copy import deepcopy
import cv2
from utils import load_and_resize_png, overlay_with_alpha

# def substitute_placeholders(text, data):
#     # Substitute passenger_reference, drive_common, and location_formulation
#     text = re.sub(r"\bpassenger_reference\b", lambda _: random.choice(data["passenger_reference"]), text)
#     text = re.sub(r"\bdrive_common\b", lambda _: random.choice(data["drive_common"]), text)
#     text = re.sub(r"\blocation_formulation\b", lambda _: random.choice(data["location_formulation"]), text)

#     # Substitute attributes like colour, fill, direction, and building
#     for key in ["colour", "fill", "direction", "building", "size"]:
#         text = re.sub(rf"\b{key}\b", lambda _: random.choice(data[key]), text)

#     return text


# def construct_hard_test(htest_sample: Dict, non_hard_test_attributes: Dict) -> List:
#     samples = []
#     # This code expect exactly two attr names
#     attr_names = list(non_hard_test_attributes.keys())
#     for attr_value_0 in non_hard_test_attributes[attr_names[0]]:
#         for attr_value_1 in non_hard_test_attributes[attr_names[1]]:
#             sample = deepcopy(htest_sample)
#             sample[attr_names[0]] = attr_value_0
#             sample[attr_names[1]] = attr_value_1
#             samples.append(sample)          
#     return samples


# def construct_train_test(train_test_dict: Dict, split: List) -> Tuple[List]:
#     train_samples = []
#     test_samples = []

#     attr_names = list(train_test_dict.keys())
#     for attr_value_0 in train_test_dict[attr_names[0]]:
#         for attr_value_1 in train_test_dict[attr_names[1]]:
#             for attr_value_2 in train_test_dict[attr_names[2]]:
#                 for attr_value_3 in train_test_dict[attr_names[3]]:
#                     sample = {}
#                     sample[attr_names[0]] = attr_value_0
#                     sample[attr_names[1]] = attr_value_1
#                     sample[attr_names[2]] = attr_value_2
#                     sample[attr_names[3]] = attr_value_3

#                     set_id = random.random()
#                     if set_id < split[0]:
#                         train_samples.append(sample)
#                     else:
#                         test_samples.append(sample)
#     return train_samples, test_samples


# def initialise_goals(self):
#     train_test_dict = {
#         "colour": hparams["colour"],
#         "building": hparams["building"],
#         "size": hparams["size"],
#         "fill": hparams["fill"]
#     }
#     non_hard_test_attributes = deepcopy(train_test_dict)
#     htest_sample = {}
#     hard_test_attributes = hparams["hard_test_attributes"]
    
#     for attr_name, attr_value in hard_test_attributes.items():
#         train_test_dict[attr_name].remove(attr_value)
#         non_hard_test_attributes.pop(attr_name)
#         htest_sample[attr_name] = attr_value
    
#     hard_test_samples = construct_hard_test(
#         htest_sample=htest_sample, 
#         non_hard_test_attributes=non_hard_test_attributes
#     )

#     split = [0.7, 0.3]
#     train_samples, test_samples = construct_train_test(
#         train_test_dict=train_test_dict, 
#         split=split
#     )


class FeatureTaxicab(gym.Env):
    def __init__(self, hparams: Dict, store_path:str, location_features: List[Dict], default_features: str, goal_pos: Tuple):
        self._grid = np.array(hparams['grid'])
        self._walls = np.equal(self._grid, 'W')
        self._location_features = location_features
        self._default_features = default_features
        self._target_location = goal_pos

        self._action_to_direction = {
            0: np.array([-1, 0]), # up
            1: np.array([1, 0]),  # down
            2: np.array([0, -1]), # left
            3: np.array([0, 1]),  # right
            4: np.array([0, 0]), # pick up passenger
            5: np.array([0, 0]) # drop off passenger
        }
        self._slip_chance = hparams['slip_chance']
        self._max_steps = hparams['max_steps']
        self._steps = 0
        self.action_space = gym.spaces.Discrete(6)

        # Seeding random generators for reproducibility
        if hparams['seed'] is not None:
            self.action_space.seed(seed=hparams['seed'])
            self.rng = np.random.default_rng(hparams['seed'])
        else:
            self.rng = np.random.default_rng()

        # [row, col, passenger_loc, has_passenger, patch_features x 4]
        self.observation_space = gym.spaces.MultiDiscrete([10] * 8)
        
        # Random start option
        self._random_start = hparams['start_pos'] == None
        if not self._random_start:
            self.start_pos = hparams['start_pos']
        self._agent_location = hparams['start_pos']
        
        self.reward = {' ': -1}
        self._store_path = store_path
        
        self._poi = None
        # Passenger spawns in one of four special feature locations, but can be dropped anywhere
        self._passenger_location = None
        self._passenger_in = 0
        _ = self.reset(options={"location_features": location_features})

    @property
    def obs(self) -> gym.spaces.MultiDiscrete:
        if self._agent_location in self._poi:
            str_features = self._poi[self._agent_location]["fature_value"]
        else:
            str_features = self._default_features
        features = [self._agent_location[0], self._agent_location[1], self._passenger_location[0], self._passenger_location[1], self._passenger_in]
        features.extend([int(f) for f in str_features])
        return np.array(features, dtype=int)
    
    @property 
    def wall_mask(self) -> np.ndarray:
        return self._walls
    
    @property
    def grid_shape(self) -> np.ndarray:
        return self._grid.shape
        
    def _pick_random_start(self):
        x = self.rng.integers(1, self.grid_shape[0])
        y = self.rng.integers(1, self.grid_shape[1])
        while self._grid[x, y] == 'W' or (x, y) == self._target_location or self._grid[x, y] == 'C' :
            x = self.rng.integers(1, self.grid_shape[0])
            y = self.rng.integers(1, self.grid_shape[1])
        return x, y

    def _init_points_of_interest(self, location_features: List[Dict]) -> Dict:
        poi = {}
        i = 0
        passenger_i = np.random.randint(0, len(location_features))
        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row, col] == 'C':
                    poi[(row, col)] = location_features[i]
                    if i == passenger_i:
                        self._passenger_location = (row, col)
                    i += 1
        return poi

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """ Reset the environment and return the initial state number
        """
        super().reset(seed=seed)
        if self._random_start:
            self._agent_location = self._pick_random_start()
        else:
            self._agent_location = self.start_pos
        self._steps = 0
        info = ''
        
        if not options or 'location_features' not in options:
            location_features = self._location_features
        else:
            location_features = options['location_features']
        self._poi = self._init_points_of_interest(location_features=location_features)
        
        assert self._grid[self._agent_location] != 'W'
        return self.obs, info

    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
            - 4: pick passenger up
            - 5: drop passenger
        """
        if isinstance(action, torch.Tensor) or isinstance(action, np.ndarray):
            action = action.item()
        assert(action >= 0)
        assert(action <= 5)
        
        # Update agent location for the movement actions
        if self.rng.random() < self._slip_chance:
            if action == 0:
                action = self.rng.choice([2, 3])
            elif action == 1:
                action = self.rng.choice([3, 2])
            elif action == 2:
                action = self.rng.choice([1, 0])
            elif action == 3:
                action = self.rng.choice([0, 1])
        
        agent_location = tuple(self.obs + self._action_to_direction[action])
        if self._grid[agent_location] != 'W':
            self._agent_location = agent_location
        assert self._grid[self._agent_location] != 'W'
        reward = self.reward[self._grid[self._agent_location]]
        
        # Handle non-movement pick up and drop passenger actions
        if action == 4:
            if self._passenger_location == self._agent_location:
                self._passenger_in = 1
            else:
                reward = -10
        elif action == 5:
            if self._agent_location == self._target_location and self._passenger_in == 1:
                reward = 20
                is_terminal = True
            else:
                reward = -10
                self._passenger_in = 0
                self._passenger_location = self._agent_location

        self._steps += 1
        info = ''
        truncated = False
        if self._max_steps is not None and self._steps >= self._max_steps:
            truncated = True
        return self.obs, reward, is_terminal, truncated, info

    def render_feature_grid(self, cell_size:int=60, use_png=False):
        """Render the grid with color fill in a vectorized manner.
        Return the upscaled color image (no text yet)."""
        rows, cols = self._grid.shape

        # 1) Initialize color array: all white
        color_arr = np.full((rows, cols, 3), fill_value=(255, 255, 255), dtype=np.uint8)

        # 2) Assign black for 'W' walls
        color_arr[self._walls] = (0, 0, 0)  # black

        # 3) Assign feature colors
        if use_png is False:
            for fkey, fdata in self._poi.items():
                color_arr[fkey] = fdata["colour"]

        # 4) Upscale each cell to cell_size x cell_size
        image = color_arr.repeat(cell_size, axis=0).repeat(cell_size, axis=1)
        return image
    
    def add_features(self, image, cell_size=60, use_png=False, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=1):
        """
        For each cell that has a feature (F0, F1, etc.):
        - If use_png=True, overlay the PNG tinted with color at 50% transparency
        - Otherwise, place the two letters in the cell
        """
        # Pre-load and resize PNG images if needed
        png_cache = {}
        text_color = (0, 0, 0)  # black text
        small_size = int(cell_size/2)

        if use_png:
            for loc, feature_dict in self._poi.items():
                if feature_dict["size"] == "big":
                    png_size = cell_size
                else:
                    png_size = small_size   
                png_cache[loc] = load_and_resize_png(feature_dict["png_path"], png_size, keep_alpha=False)
                png_dir = os.path.dirname(feature_dict["png_path"])
            agent_image = load_and_resize_png(os.path.join(png_dir, "taxi.png"), small_size, keep_alpha=True)
            passenger_image = load_and_resize_png(os.path.join(png_dir, "passenger.png"), small_size, keep_alpha=True)
            
        # Iterate over each feature type
        for loc, feature_dict in self._poi.items():
            letters = feature_dict["feature_value"]
            r = loc[0]
            c = loc[1]

            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size

            if use_png:
                if feature_dict["size"] == "big":
                    image[y0:y1, x0:x1] = png_cache[loc]
                else:
                    x_offset = x0 + (cell_size - png_size) // 2
                    y_offset = y0 + (cell_size - png_size) // 2
                    image[y_offset:y_offset+png_size, x_offset:x_offset+png_size] = png_cache[loc]
            else:
                # Place four letters in a 2×2 arrangement, shifted left/down.
                # 'quarter' is 1/4 of the cell size; 
                # these define approximate "centers" for each quadrant.
                quarter = cell_size // 4

                # Define small shifts (in pixels)
                # negative x_shift => moves text to the left
                # positive y_shift => moves text down
                x_shift, y_shift = 5, 5

                # Quadrant centers (x0, y0 is the top-left corner of the cell):
                # Then we shift them slightly left (subtract from x) and down (add to y).
                pos1 = (x0 + quarter - x_shift,      y0 + quarter + y_shift)        # top-left
                pos2 = (x0 + 3 * quarter - x_shift,  y0 + quarter + y_shift)        # top-right
                pos3 = (x0 + quarter - x_shift,      y0 + 3 * quarter + y_shift)    # bottom-left
                pos4 = (x0 + 3 * quarter - x_shift,  y0 + 3 * quarter + y_shift)    # bottom-right

                cv2.putText(image, letters[0], pos1, font, font_scale,
                            text_color, thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, letters[1], pos2, font, font_scale,
                            text_color, thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, letters[2], pos3, font, font_scale,
                            text_color, thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, letters[3], pos4, font, font_scale,
                            text_color, thickness, lineType=cv2.LINE_AA)

        # Plot current position of taxicab and passenger
        if use_png:
            for loc, overlay_image in zip([self._agent_location, self._passenger_location], [agent_image, passenger_image]):
                y0, y1 = loc[0] * cell_size, (loc[0] + 1) * cell_size
                x0, x1 = loc[1] * cell_size, (loc[1] + 1) * cell_size
                x_offset = x0 + (cell_size - small_size) // 2
                y_offset = y0 + (cell_size - small_size) // 2
                overlay_with_alpha(image, overlay_image, x_offset, y_offset)
        else:
            for loc, symbol in zip([self._agent_location, self._passenger_location], ['T', 'P']):
                y0, y1 = loc[0] * cell_size, (loc[0] + 1) * cell_size
                x0, x1 = loc[1] * cell_size, (loc[1] + 1) * cell_size
                cell_center = (x0 + cell_size // 2, y0 + cell_size // 2)
                cv2.putText(image, symbol, cell_center, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    def store_frame(self, plot_name:str='table') -> None:
        image = self.render_feature_grid(use_png=True)
        self.add_features(image=image, use_png=True)
        output_path = os.path.join(self._store_path, f'{plot_name}_features.png')
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    from utils import setup_artefact_paths

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path)
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
    location_features = [
        {
            "colour": (0, 0, 255),
            "building": "hospital",
            "size": "big",
            "fill": "filled",
            "feature_value": "3111",
            "png_path": "/home/djordje/dev/rlexp/envs/taxicab/assets/red_hospital_filled.png"
        },
        {
            "colour": (255, 0, 0),
            "building": "school",
            "size": "big",
            "fill": "filled",
            "feature_value": "1211",
            "png_path": "/home/djordje/dev/rlexp/envs/taxicab/assets/blue_school_filled.png"
        },
        {
            "colour": (0, 255, 255),
            "building": "library",
            "size": "big",
            "fill": "filled",
            "feature_value": "4312",
            "png_path": "/home/djordje/dev/rlexp/envs/taxicab/assets/yellow_library_filled.png"
        },
        {
            "colour": (0, 255, 0),
            "building": "office",
            "size": "small",
            "fill": "outlined",
            "feature_value": "3421",
            "png_path": "/home/djordje/dev/rlexp/envs/taxicab/assets/green_office_outlined.png"
        }
    ]

    env = FeatureTaxicab(
        hparams=hparams,
        location_features=location_features,
        default_features="0000",
        goal_pos=(1,1),
        store_path=store_path
    )
    env.store_frame()
    
    # for episode in tqdm(range(hparams['n_episodes'])):
    #     obs, _ = env.reset()
    #     done = False

    #     while not done:
    #         action = env.action_space.sample()
    #         next_obs, reward, terminated, truncated, _ = env.step(action)
    #         obs = next_obs
    #         done = terminated or truncated
