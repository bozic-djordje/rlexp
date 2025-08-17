from abc import abstractmethod
from typing import Dict, List, Optional, Tuple
import os
from copy import deepcopy
import numpy as np
import torch
import gymnasium as gym
import cv2
from utils import load_and_resize_png, overlay_with_alpha


ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DEFAULT_OBJECTS = [
        {
            "loc": (1,1),
            "shape": "ball",
            "colour": "red",
            "is_goal": True
        },
        {
            "loc": (1, 8),
            "shape": "diamond",
            "colour": "blue"
        }
    ]


class Shapes(gym.Env):
    def __init__(self, objects: List[Dict], grid: List, feature_order: List, features: Dict, store_path:str, default_feature:int=0, max_steps:int=200, slip_chance:float=0, seed:int=0, goal_channel:bool=False, obs_type:str="box"):
        self._store_path = store_path
        self._assets_path = ASSETS_PATH
        
        self._feature_order = feature_order
        self._features = features
        self._feature_map, self._feature_rmap = self._init_feature_map()
        self._default_feature = default_feature

        self._grid = np.array(grid)
        self._objects = objects
        self._slip_chance = slip_chance
        
        self._action_to_direction = {
            0: np.array([-1, 0]), # up
            1: np.array([1, 0]),  # down
            2: np.array([0, -1]), # left
            3: np.array([0, 1])  # right
        }

        self._action_to_str = {
            0: "up",
            1: "down",
            2: "left",
            3: "right"
        }

        self._max_steps = max_steps
        self._steps = 0

        self.action_space = gym.spaces.Discrete(4)

        # Seeding random generators for reproducibility
       
        self.action_space.seed(seed=seed)
        self.rng = np.random.default_rng(seed)

        # [channels, height, width]
        # channels = (agent_present, shape_feature, colour_feature)
        # or
        # channels = (agent, goal, shape_feature, colour_feature)
        self._obs_type = obs_type
        self._goal_channel = goal_channel
        self._first_feature_ind = 1 + int(goal_channel)
        self._num_channels = len(self._feature_order) + 1 + int(goal_channel)
        self.observation_space = self._init_observation_space()
        
        self._game_map = None
        self._game_vec = None
        self._goal_location = None
        self._agent_location = None
        self._agent_orientation = None
        _ = self.reset(options={"objects": objects})

    def _init_feature_map(self) -> Dict:
        feature_map = {}
        feature_rmap = {}
        i = 0

        for feature_name in self._feature_order:
            values = self._features[feature_name]
            feature_rmap[i] = {}
            
            for ind, name in enumerate(values):
                feature_map[name] = ind+1
                feature_rmap[i][ind+1] = name

            i+=1
        return feature_map, feature_rmap
    
    def _init_game_map(self):
        game_map = np.zeros((self._num_channels, self._grid.shape[0], self._grid.shape[1]), dtype=np.int8)
        walls = np.equal(self._grid, 'W')
        game_map[:, walls] = -1

        num_features = len(self._feature_order)
        vec_len = len(self._objects) * (2 + num_features) + 2
        vec_len += num_features if self._goal_channel else 0
        game_vec = np.ones(vec_len, dtype=np.int8) * -1
        
        goal_location = None
        goal_features = np.ones(num_features, dtype=np.int8) * -1
        
        agent_location = self._init_start_location()
        game_map[0, agent_location[0], agent_location[1]] = 1
        game_vec[0:2] = agent_location
        vec_ind = 2

        obj_cpy = deepcopy(self._objects)
        for obj in obj_cpy:
            loc = obj.pop("loc")
            is_goal = obj.pop("is_goal", False)
            if is_goal:
                goal_location = loc
            
            game_vec[vec_ind:vec_ind+2] = loc
            vec_ind += 2
            
            for i, (feature, value) in enumerate(obj.items()):
                channel_index = self._feature_order.index(feature) + self._first_feature_ind
                index = (channel_index,) + loc
                game_map[index] = self._feature_map[value]
                game_vec[vec_ind] = self._feature_map[value]
                vec_ind += 1

                if is_goal:
                    goal_features[i] = self._feature_map[value]

        if self._goal_channel:
            game_map[1, goal_location[0], goal_location[1]] = 1
            game_vec[-num_features:] = goal_features
        
        assert not np.any(game_vec == -1)
        assert not np.any(goal_features == -1)
        return game_map, game_vec, goal_location, agent_location
    
    def _init_start_location(self):
        specified_locs = np.where(self._grid == 'A')
        candidates = list(zip(*specified_locs))
        if len(candidates) == 0:
            empty_locations = np.where(self._grid == ' ')
            candidates = list(zip(*empty_locations))
        loc = self.rng.choice(candidates)
        return tuple(loc)
    
    def _init_observation_space(self) -> gym.spaces.Space:
        if self._obs_type == "vec":
            vec_len = len(self._objects) * (2 + len(self._feature_order)) + 2
            vec_len += 2 if self._goal_channel else 0
            # [agent_x, agent_y, obj_1_x, obj_1_y, obj_1_feature_1, obj_1_feature_2, ..., <goal_x>, <goal_y>]
            obs_space = gym.spaces.MultiDiscrete([10] * vec_len)
        elif self._obs_type == "box":
            obs_space = gym.spaces.Box(
                low=-1,
                high=6,
                shape=(self._num_channels, 3, 3),
                dtype=np.int8
            )
        else:
            raise ValueError(f"Obs type must be either vec or box. Unrecognised type {self._obs_type}.")

        return obs_space
    
    @property
    def obs(self) -> gym.spaces.Box:
        if self._obs_type == "vec":
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            obs[0:2] = self.agent_location
            obs[2:] = self._game_vec[2:]
        else:
            obs = self._game_map[:, self.agent_location[0]-1:self.agent_location[0]+2, 
                                 self.agent_location[1]-1:self.agent_location[1]+2]
        return obs
    
    @property 
    def wall_mask(self) -> np.ndarray:
        return np.equal(self._grid, 'W')
    
    @property
    def grid_shape(self) -> np.ndarray:
        return self._grid.shape
    
    @property
    def agent_location(self) -> Tuple:
        return tuple(self._agent_location)
    
    def reset(self, seed: Optional[int]=None, options: Optional[dict]={}):
        """ Reset the environment and return the initial state number
        """
        super().reset(seed=seed)
        self._steps = 0
        info = {}

        objects = options.get("objects", None)
        if objects is not None:
            self._objects = objects
        
        self._game_map, self._game_vec, self._goal_location, self._agent_location = self._init_game_map()
        self._agent_orientation = 3
   
        assert self._grid[self._agent_location] != 'W'
        return self.obs, info
    
    def _movement(self, action) -> None:
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
            - 4: pick up
            - 5: drop
        """
        if isinstance(action, torch.Tensor) or isinstance(action, np.ndarray):
            action = action.item()
        assert(action >= 0)
        assert(action <= 5)
        self._game_map[0, self.agent_location[0], self.agent_location[1]] = 0

        if action < 4:
            self._agent_orientation = action

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
        
        agent_location = tuple(list(self._agent_location) + self._action_to_direction[action])
        if self._grid[agent_location] != 'W':
            self._agent_location = agent_location
        assert self._grid[self._agent_location] != 'W'
        self._game_map[0, self.agent_location[0], self.agent_location[1]] = 1
        
        non_walls = ~np.equal(self._grid, 'W')
        assert(sum(self._game_map[0, non_walls]) == 1)

    @abstractmethod
    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
            - 4: pick up #not used for now
            - 5: drop #not used for now
        """
        self._movement(action=action)
        self._steps += 1
        info = {}
        truncated = False
        if self._max_steps is not None and self._steps >= self._max_steps:
            truncated = True
        
        # Define these in subclass
        reward = None
        is_terminal = None
        return self.obs, reward, is_terminal, truncated, info
    
    def _get_asset_path(self, features: List) -> str:
        # Features with relevant assets have non zero values
        if sum(features) < len(features):
            return None
        
        asset_name = ""
        for channel_ind, feature_value in enumerate(features):
            asset_name += f"{self._feature_rmap[channel_ind][feature_value]}_"
        
        asset_name = f"{asset_name[:-1]}.png"
        asset_path = os.path.join(ASSETS_PATH, asset_name)
        return asset_path
    
    def _get_agent_asset_path(self) -> str:
        asset_name = f"agent_{self._action_to_str[self._agent_orientation]}.png"
        asset_path = os.path.join(ASSETS_PATH, asset_name)
        return asset_path
    
    def _get_goal_asset_path(self) -> str:
        asset_path = os.path.join(ASSETS_PATH, "goal.png")
        return asset_path

    def _render_feature_grid(self, cell_size:int=60):
        """Render the grid with color fill in a vectorized manner.
        Return the upscaled color image (no text yet)."""
        rows, cols = self._grid.shape

        # 1) Initialize color array: all white
        color_arr = np.full((rows, cols, 3), fill_value=(255, 255, 255), dtype=np.uint8)

        # 2) Assign black for 'W' walls
        color_arr[self.wall_mask] = (0, 0, 0)  # black

        # 4) Upscale each cell to cell_size x cell_size
        image = color_arr.repeat(cell_size, axis=0).repeat(cell_size, axis=1)
        return image
    
    def _add_features(self, image, cell_size=60):
        """
        For each cell that has a feature (F0, F1, etc.):
        - Otherwise, place the two letters in the cell
        """
        # Pre-load and resize PNG images if needed
        png_cache = {}
        small_size = cell_size

        agent_image = load_and_resize_png(
            path=self._get_agent_asset_path(), 
            cell_size=cell_size,
            keep_alpha=True
        )

        goal_image = load_and_resize_png(
            path=self._get_goal_asset_path(),
            cell_size=cell_size,
            keep_alpha=True
        )

        for r in range(self._game_map.shape[1]):
            for c in range(self._game_map.shape[2]):
                features = list(self._game_map[self._first_feature_ind:, r, c])
                asset_path = self._get_asset_path(features=features)
                if asset_path is not None:
                    png_cache[(r,c)] = load_and_resize_png(
                        path=asset_path, 
                        cell_size=cell_size, 
                        keep_alpha=False
                    )
        
        # Iterate over each feature type
        for loc, val in png_cache.items():
            r = loc[0]
            c = loc[1]

            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            image[y0:y1, x0:x1] = val

        # Plot current agent position
        y0, y1 = self._agent_location[0] * cell_size, (self._agent_location[0] + 1) * cell_size
        x0, x1 = self._agent_location[1] * cell_size, (self._agent_location[1] + 1) * cell_size
        x_offset = x0 + (cell_size - small_size) // 2
        y_offset = y0 + (cell_size - small_size) // 2
        overlay_with_alpha(image, agent_image, x_offset, y_offset)
        
        # Plot goal position
        y0, y1 = self._goal_location[0] * cell_size, (self._goal_location[0] + 1) * cell_size
        x0, x1 = self._goal_location[1] * cell_size, (self._goal_location[1] + 1) * cell_size
        x_offset = x0 + (cell_size - small_size) // 2
        y_offset = y0 + (cell_size - small_size) // 2
        overlay_with_alpha(image, goal_image, x_offset, y_offset)
    
    def render_frame(self) -> np.ndarray:
        image = self._render_feature_grid()
        self._add_features(image=image)
        return image

    def store_frame(self, plot_name:str='table') -> None:
        image = self.render_frame()
        output_path = os.path.join(self._store_path, f'{plot_name}.png')
        cv2.imwrite(output_path, image)


class ShapesGoto(Shapes):
    def step(self, action):
        # Superclass will perform the agent movement but will not do any of the additional actions
        # or provide sensible reward
        obs, _, _, truncated, info = super().step(action)
        
        confounder_locations = set([obj["loc"] for obj in self._objects if "is_goal" not in obj or not obj["is_goal"]])
        is_terminal = False
        reward = -1

        if self._agent_location == self._goal_location:
            is_terminal = True
            reward = 10
        elif self._agent_location in confounder_locations:
            reward = -10
        
        return obs, reward, is_terminal, truncated, info
    

# TODO: You do not get negative rewards when you step on an object that isn't goal object
class ShapesGotoEasy(Shapes):
    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        
        is_terminal = False
        reward = -1

        if self._agent_location == self._goal_location:
            is_terminal = True
            reward = 10
        
        return obs, reward, is_terminal, truncated, info


# TODO
class ShapesPickup(Shapes):
    def step(self, action):
        return super().step(action)


# TODO: Equivalent to Taxicab
class ShapesRetrieve(Shapes):
    def step(self, action):
        return super().step(action)


if __name__ == '__main__':
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, config_name="shapes")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    grid = hparams["grid"]
    feature_order = hparams["use_features"]
    features = hparams["features"]
    
    for key in features.keys():
        if key not in set(feature_order):
            for obj in DEFAULT_OBJECTS:
                obj.pop(key)

    env = ShapesGoto(
        objects=DEFAULT_OBJECTS,
        grid=grid,
        feature_order=feature_order,
        features=features,
        store_path=store_path,
        goal_channel=True,
        obs_type="vec"
    )
    env.store_frame()
    
    i = 0
    for episode in tqdm(range(3)):
        obs, _ = env.reset(options={})
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
        env.store_frame(plot_name=f"final_step_task_{i}")
        i+= 1