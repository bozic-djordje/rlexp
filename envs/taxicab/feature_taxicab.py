from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import gymnasium as gym
import cv2
from utils import load_and_resize_png, overlay_with_alpha, COLOUR_MAP


ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DEFAULT_FEATURES = [
    { "colour": "red",    "building": "hospital"},
    { "colour": "blue",   "building": "school"  },
    { "colour": "yellow", "building": "library" },
    { "colour": "green",  "building": "office"  },
]

class FeatureTaxicab(gym.Env):
    def __init__(self, hparams: Dict, location_features: List[Dict], store_path:str, origin_ind:int=None, dest_ind:int=None, easy_mode:bool=False, goto_mode:bool=False):
        self._store_path = store_path
        self._assets_path = ASSETS_PATH
        # If agent is penalised for wrong passanger drop off location
        self._easy_mode = easy_mode
        # If the cab already has the passenger in at the start. easy_mode encapsulates goto_mode.
        self._goto_mode = goto_mode or easy_mode
        
        self._feature_map = {}
        self._feature_order = hparams["attribute_order"]
        for feature_name in self._feature_order:
            self._feature_map[feature_name] = hparams[feature_name]
        self._default_features = [int(hparams["default_feature"]) for _ in range(len(self._feature_order))]

        self._grid = np.array(hparams['grid'])
        self._walls = np.equal(self._grid, 'W')
        self._location_features = self._assign_feature_values(location_features=location_features)
        self.pomdp = hparams["pomdp"]
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
        self.observation_space = gym.spaces.MultiDiscrete([10] * 9)
        
        # Random start option
        self._random_start = hparams['start_pos'] == None
        if not self._random_start:
            self.start_pos = hparams['start_pos']
        self._agent_location = hparams['start_pos']
        
        self._poi = None
        # Passenger spawns in one of four special feature locations
        self._passenger_location = None
        self._destination_location = None
        self._passenger_in = 0
        
        _ = self.reset(options={"location_features": self._location_features, "origin_ind": origin_ind, "dest_ind": dest_ind})

    @property
    def obs(self) -> gym.spaces.MultiDiscrete:
        features = [self._agent_location[0], self._agent_location[1], self._passenger_in]
        if not self.pomdp:
            origin_dest_features = [self._passenger_location[0], self._passenger_location[1], self._destination_location[0], self._destination_location[1]]
        else:
            # TODO: Should we just add zeros, or add nothing at all?
            origin_dest_features = [0, 0, 0, 0]
        features.extend(origin_dest_features)

        if self._agent_location in self._poi:
            obs_features = self._poi[self._agent_location]["feature_value_list"]
        else:
            obs_features = self._default_features
        features.extend(obs_features)
        return np.array(features, dtype=int)
    
    @property 
    def wall_mask(self) -> np.ndarray:
        return self._walls
    
    @property
    def grid_shape(self) -> np.ndarray:
        return self._grid.shape
    
    def _generate_poi_indices(self) -> Tuple[int]:
        origin_ind = self.rng.integers(0, 4)
        dest_ind = origin_ind
        while dest_ind == origin_ind:
            dest_ind = self.rng.integers(0, 4)
        assert(origin_ind != dest_ind)
        return origin_ind, dest_ind
        
    def _assign_feature_values(self, location_features:List[Dict]) -> Tuple[List, str]:
        for attr_dict in location_features:
            feature_value_str = ""
            feature_value_lst = []
            feature_asset_name = ""
            for feature_name in self._feature_map.keys():
                feature_text = attr_dict[feature_name]
                feature_value = self._feature_map[feature_name].index(feature_text) + 1
                feature_value_str += str(feature_value)
                feature_value_lst.append(feature_value)
                feature_asset_name += f'{feature_text}_'
            # Hack for compatibility with legacy asset names where filled was one of the features
            feature_asset_name += "filled"
            
            # Feature values stored as a string
            attr_dict["feature_value_text"] = feature_value_str
            # Feature values stored as an array of integers
            attr_dict["feature_value_list"] = feature_value_lst
            attr_dict["colour_code"] = COLOUR_MAP[attr_dict["colour"]]
            attr_dict["png_path"] = os.path.join(self._assets_path, f'{feature_asset_name}.png')
        return location_features

    def _pick_random_start(self):
        x = self.rng.integers(1, self.grid_shape[0])
        y = self.rng.integers(1, self.grid_shape[1])
        while self._grid[x, y] == 'W' or (x, y) == self._destination_location or self._grid[x, y] == 'C' :
            x = self.rng.integers(1, self.grid_shape[0])
            y = self.rng.integers(1, self.grid_shape[1])
        return x, y

    def _init_points_of_interest(self, location_features: List[Dict], origin_ind:int, dest_ind:int) -> Dict:
        poi = {}
        i = 0
        check = 0

        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row, col] == 'C':
                    poi[(row, col)] = location_features[i]
                    if i == origin_ind:
                        self._passenger_location = (row, col)
                        check += 1
                    if i == dest_ind:
                        self._destination_location = (row, col)
                        check += 1
                    i += 1
        assert(check == 2)
        assert(self._destination_location != self._passenger_location)
        return poi

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """ Reset the environment and return the initial state number
        """
        super().reset(seed=seed)
        self._steps = 0
        info = {}
        # By default we generate random origin and destination locations
        origin_ind, dest_ind = self._generate_poi_indices()

        # If the reset was called with options then we update env parameters
        # before initialising points of interest. This corresponds to the new task!
        # If origin and destination locations are passed in options, we use those
        if options and 'location_features' in options:
            self._location_features = self._assign_feature_values(location_features=options['location_features'])    
            if 'origin_ind' in options and options['origin_ind'] is not None:
                origin_ind = options['origin_ind']
            if 'dest_ind' in options and options['dest_ind'] is not None:
                dest_ind = options['dest_ind']
            
        self._poi = self._init_points_of_interest(
            location_features=self._location_features,
             origin_ind=origin_ind, 
             dest_ind=dest_ind
        )
        
        if self._random_start:
            self._agent_location = self._pick_random_start()
        else:
            self._agent_location = self.start_pos
        
        if self._goto_mode:
            self._passenger_in = 1
            self._passenger_location = self._agent_location

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
        is_terminal = False
        reward = -1

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
        
        agent_location = tuple(self.obs[0:2] + self._action_to_direction[action])
        if self._grid[agent_location] != 'W':
            self._agent_location = agent_location
        assert self._grid[self._agent_location] != 'W'
        
        # Handle non-movement pick up and drop passenger actions
        if action == 4 and not self._goto_mode:
            if self._passenger_in == 0 and self._passenger_location == self._agent_location:
                self._passenger_in = 1
            else:
                reward = -10
        elif action == 5:
            # The episode ends only if passenger is dropped off to the correct location
            # Otherwise agent gets -10 penalty
            if self._passenger_in == 1 and self._agent_location == self._destination_location:
                self._passenger_in = 0
                is_terminal = True
                reward = 20
            else:
                # We only penalise wrong drop off to the location which is among the points of interest
                if not self._easy_mode and self._agent_location in self._poi.keys():
                    reward = -10
        
        if self._passenger_in:
            self._passenger_location = self._agent_location

        self._steps += 1
        info = {}
        truncated = False
        if self._max_steps is not None and self._steps >= self._max_steps:
            truncated = True
        return self.obs, reward, is_terminal, truncated, info

    def _render_feature_grid(self, cell_size:int=60, use_png=True):
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
                color_arr[fkey] = fdata["colour_code"]

        # 4) Upscale each cell to cell_size x cell_size
        image = color_arr.repeat(cell_size, axis=0).repeat(cell_size, axis=1)
        return image
    
    def _add_features(self, image, cell_size=60, use_png=True, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=1):
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
                png_cache[loc] = load_and_resize_png(feature_dict["png_path"], cell_size, keep_alpha=False)
                png_dir = os.path.dirname(feature_dict["png_path"])
            agent_image = load_and_resize_png(os.path.join(png_dir, "taxi.png"), small_size, keep_alpha=True)
            passenger_image = load_and_resize_png(os.path.join(png_dir, "passenger.png"), small_size, keep_alpha=True)
            destination_image = load_and_resize_png(os.path.join(png_dir, "destination.png"), small_size, keep_alpha=True)
            
        # Iterate over each feature type
        for loc, feature_dict in self._poi.items():
            letters = feature_dict["feature_value_text"]
            r = loc[0]
            c = loc[1]

            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size

            if use_png:
                image[y0:y1, x0:x1] = png_cache[loc]
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
                # pos3 = (x0 + quarter - x_shift,      y0 + 3 * quarter + y_shift)    # bottom-left
                # pos4 = (x0 + 3 * quarter - x_shift,  y0 + 3 * quarter + y_shift)    # bottom-right

                cv2.putText(image, letters[0], pos1, font, font_scale,
                            text_color, thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, letters[1], pos2, font, font_scale,
                            text_color, thickness, lineType=cv2.LINE_AA)
                # cv2.putText(image, letters[2], pos3, font, font_scale,
                #             text_color, thickness, lineType=cv2.LINE_AA)
                # cv2.putText(image, letters[3], pos4, font, font_scale,
                #             text_color, thickness, lineType=cv2.LINE_AA)

        # Plot current position of taxicab and passenger
        if use_png:
            for loc, overlay_image in zip([self._agent_location, self._passenger_location], [agent_image, passenger_image]):
                y0, y1 = loc[0] * cell_size, (loc[0] + 1) * cell_size
                x0, x1 = loc[1] * cell_size, (loc[1] + 1) * cell_size
                x_offset = x0 + (cell_size - small_size) // 2
                y_offset = y0 + (cell_size - small_size) // 2
                overlay_with_alpha(image, overlay_image, x_offset, y_offset)
            
            y0, y1 = self._destination_location[0] * cell_size, (self._destination_location[0] + 1) * cell_size
            x0, x1 = self._destination_location[1] * cell_size, (self._destination_location[1] + 1) * cell_size
            x_offset = x0 + (cell_size - small_size) // 2
            y_offset = y0 + (cell_size - small_size) // 2
            overlay_with_alpha(image, destination_image, x_offset, y_offset)
            
        else:
            for loc, symbol in zip([self._agent_location, self._passenger_location, self._destination_location], ['T', 'P', 'D']):
                y0, y1 = loc[0] * cell_size, (loc[0] + 1) * cell_size
                x0, x1 = loc[1] * cell_size, (loc[1] + 1) * cell_size
                cell_center = (x0 + cell_size // 2, y0 + cell_size // 2)
                cv2.putText(image, symbol, cell_center, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    
    def render_frame(self, use_png:bool=True) -> np.ndarray:
        image = self._render_feature_grid(use_png=use_png)
        self._add_features(image=image, use_png=use_png)
        return image

    def store_frame(self, plot_name:str='table', use_png:bool=True) -> None:
        image = self.render_frame(use_png=use_png)
        output_path = os.path.join(self._store_path, f'{plot_name}.png')
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, config_name="taxicab")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    env = FeatureTaxicab(
        hparams=hparams,
        location_features=DEFAULT_FEATURES,
        origin_ind=1,
        dest_ind=2,
        store_path=store_path
    )
    env.store_frame(use_png=True)
    i = 0
    for episode in tqdm(range(hparams['n_episodes'])):
        origin_ind = np.random.randint(0, len(DEFAULT_FEATURES))
        dest_ind = (origin_ind + 1) % len(DEFAULT_FEATURES)
        options = {
            "location_features": DEFAULT_FEATURES,
            "origin_ind": origin_ind, 
            "dest_ind": dest_ind
        }
        obs, _ = env.reset(options=options)
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
        env.store_frame(plot_name=f"final_step_task_{i}", use_png=True)
        i+= 1
