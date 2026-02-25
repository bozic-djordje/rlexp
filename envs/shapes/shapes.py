from abc import abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union
import os
from copy import deepcopy
import numpy as np
import torch
import gymnasium as gym
import cv2
from utils import load_and_resize_png, overlay_with_alpha


ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
RND_FTR_RNG = 2


class SalientObj:
    rng = np.random.default_rng(RND_FTR_RNG)

    def __init__(self, is_goal:bool=False):
        self._loc = None
        self._id = None
        self._ftr_name_to_val = None
        self.cnfd_ftrs = {}
        self.is_goal = is_goal

        # TODO: At some point in the future implement string redouts of events that have happened.
        # This OO-MDP factored representation is really suitable for it. OO-MDPs even have effects!
        # For example: "You have picked up blue ball", "You have unlocked the door", ...
        self.effects: List[str] = []
    
    @property
    def placed(self):
        return self._loc is not None
    
    @property
    def loc(self):
        return tuple(self._loc)
    
    @property
    def unique_id(self):
        return self._id
    
    def activate(self, loc: Tuple, unique_id: int, ftr_name_to_val: Dict, n_confound_ftrs:int=0):
        self._loc = loc
        self._id = unique_id
        self._ftr_name_to_val = ftr_name_to_val
        self.cnfd_ftrs = {f"random_feature_{i}": SalientObj.rng.integers(0, RND_FTR_RNG) 
                                 for i in range(n_confound_ftrs)}

    @property
    @abstractmethod
    def asset_path(self):
        pass


class Shape(SalientObj):
    def __init__(self, shape, colour, is_goal:bool=False):
        super().__init__(is_goal=is_goal)
        self.shape = shape
        self.colour = colour
        self._picked_up = False

    def __eq__(self, value):
        return self.shape == value.shape and self.colour == value.colour
    
    def __str__(self):
        description = f"{self.colour} {self.shape}"
        if self._picked_up:
            description += " picked up"
        return description
    
    @property
    def colour_feature(self):
        return self._ftr_name_to_val[self.colour]
    
    @property
    def shape_feature(self):
        return self._ftr_name_to_val[self.shape]
    
    @property
    def is_key(self):
        return self.shape == 'key'
    
    @property
    def picked_up(self):
        return self._picked_up
    
    def pick_up(self):
        self._picked_up = True

    def drop(self, loc):
        self._picked_up = False
        self._loc = loc
    
    def move(self, loc):
        self._loc = loc
    
    def set_goal(self):
        self.is_goal = True
    
    @property
    def asset_path(self):
        return os.path.join(ASSETS_PATH, f"{self.shape}_{self.colour}.png")
        

class Door(SalientObj):
    def __init__(self, colour, is_goal:bool=False, locked:bool=True):
        super().__init__(is_goal=is_goal)
        self.colour = colour
        self._locked = locked

    def __eq__(self, value):
        return self.colour == value.colour
    
    def __str__(self):
        description = f"{self.colour} door"
        if self._locked:
            description = "locked " + description
        else:
            description = "unlocked " + description
        return description
    
    @property
    def colour_feature(self):
        return self._ftr_name_to_val[self.colour]
    
    @property
    def locked(self):
        return self._locked

    def use_key(self):
        self._locked = not self._locked

    @property
    def asset_path(self):
        closed = "closed" if self._locked else "open"
        return os.path.join(ASSETS_PATH, f"{closed}_{self.colour}.png")
    

class Actor(SalientObj):
    def __init__(self, loc):
        super().__init__()
        self._id = 0
        self._loc = loc
        self._last_mov = "up"
        self._inventory = None

    def set_last_move(self, act: str):
        self._last_mov = act
    
    def pick_up(self, obj: Shape) -> bool:
        if self._inventory is None:
            self._inventory = obj
            self._inventory.pick_up()
            return True
        else:
            return False
    
    def drop(self) -> bool:
        if self._inventory is not None:
            self._inventory.drop(loc=self._loc)
            self._inventory = None
            return True
        else:
            return False
        
    def move(self, loc: Tuple, act: str):
        self._loc = loc
        self.set_last_move(act)
        if self._inventory is not None:
            self._inventory.move(loc=loc)
    
    @property
    def inventory(self):
        return self._inventory
    
    @property
    def loc(self):
        return self._loc
    
    @property
    def asset_path(self):
        return os.path.join(ASSETS_PATH, f"agent_{self._last_mov}.png")


class Wall:
    def __init__(self, loc):
        self._loc = loc


DEFAULT_OBJECTS = [
    Shape(shape="ball", colour="blue", is_goal=True),
    Shape(shape="key", colour="red"),
    Shape(shape="key", colour="blue"),
    Shape(shape="ball", colour="red")
    ]

DEFAULT_DOORS = [
    Door(colour="blue"),
    Door(colour="red")
]


class GameMap:
    def __init__(self, grid: List[str], objects: List[Shape], doors: List[Door], features: Dict, n_confound_ftrs:int=0, one_hot:bool=True, seed:int=0):
        self.rng = np.random.default_rng(seed)
        
        self._walls = []
        self._objects: List[Shape] = deepcopy(objects)
        self._doors: List[Door] = deepcopy(doors)
        self._door_locs: List[Tuple] = []

        self._one_hot = one_hot

        grid_mat = np.array(grid)
        self._wall_mask = np.equal(grid_mat, 'W')
        self._door_mask = np.equal(grid_mat, 'D')

        # Map between feature names (e.g. blue) and their values in the observation space (e.g. 2)
        self.ftr_name_to_val = {}
        # Map between feature category (e.g. shape) and number of possible shapes. Door is not considered a shape.
        self.ftr_category_to_range = {}
        
        for feature_category, feature_names in features.items():
            self.ftr_category_to_range[feature_category] = len(feature_names)
            if "door" in feature_names:
                self.ftr_category_to_range[feature_category] -= 1
            
            for idx, name in enumerate(feature_names):
                self.ftr_name_to_val[name] = idx+1

        # repr_vec - a factored state observed by the agent
        self.n_cnfd_ftrs = n_confound_ftrs
        self.n_obj_ftrs = 5 + n_confound_ftrs
        self.n_door_ftrs = 4 + n_confound_ftrs
        self.n_agent_ftrs = 2

        self._obs = np.zeros(self.n_obj_ftrs*len(objects) + self.n_door_ftrs*len(doors) + self.n_agent_ftrs, dtype=np.uint8)
        self._door_idxs_start = self.n_obj_ftrs*len(objects)
        self._agent_idxs_start = self.n_obj_ftrs*len(objects) + self.n_door_ftrs*len(doors)
        self._n_objects = len(objects)
        self._n_doors = len(doors)

        if self._one_hot:
            if "colour" not in self.ftr_category_to_range or "shape" not in self.ftr_category_to_range:
                raise ValueError("One-hot observations require both 'colour' and 'shape' feature categories.")

            self._colour_ftr_range = self.ftr_category_to_range["colour"]
            self._shape_ftr_range = self.ftr_category_to_range["shape"]
            self._colour_eye = np.eye(self._colour_ftr_range, dtype=np.uint8)
            self._shape_eye = np.eye(self._shape_ftr_range, dtype=np.uint8)
            self._rnd_ftr_eye = np.eye(RND_FTR_RNG, dtype=np.uint8)
        
        obj_idx = 0
        door_idx = 0
        
        self._actor: Actor = None
        self._goal_object: Union[Shape, Door] = None
        self._terminal_map_loc = None
        
        for x in range(len(grid)):
            row = grid[x]
            for y in range(len(row)):
                elem = grid[x][y]
                if elem == 'W':
                    self._walls.append(Wall(loc=(x,y)))
                elif elem == ' ':
                    pass
                elif elem == 'A':
                    self._actor = Actor(loc=(x,y))
                    self._obs[self._agent_idxs_start] = x
                    self._obs[self._agent_idxs_start+1] = y
                elif elem == 'O' or elem == 'K':
                    obj = self._objects[obj_idx]
                    obj.activate(loc=(x,y), unique_id=obj_idx, ftr_name_to_val=self.ftr_name_to_val, n_confound_ftrs=n_confound_ftrs)
                    self._obs[obj_idx*self.n_obj_ftrs] = x
                    self._obs[obj_idx*self.n_obj_ftrs+1] = y
                    self._obs[obj_idx*self.n_obj_ftrs+2] = obj.colour_feature
                    self._obs[obj_idx*self.n_obj_ftrs+3] = obj.shape_feature
                    self._obs[obj_idx*self.n_obj_ftrs+4] = int(obj.picked_up)

                    for i in range(n_confound_ftrs):
                        self._obs[obj_idx*self.n_obj_ftrs+5+i] = obj.cnfd_ftrs[f"random_feature_{i}"]
                    
                    if obj.is_goal:
                        self._goal_object = obj
                    
                    obj_idx += 1
                elif elem == 'D':
                    door = self._doors[door_idx]
                    door.activate(loc=(x,y), unique_id=door_idx, ftr_name_to_val=self.ftr_name_to_val, n_confound_ftrs=n_confound_ftrs)
                    self._door_locs.append((x,y))
                    self._obs[self._door_idxs_start + door_idx*self.n_door_ftrs] = x
                    self._obs[self._door_idxs_start + door_idx*self.n_door_ftrs+1] = y
                    self._obs[self._door_idxs_start + door_idx*self.n_door_ftrs+2] = door.colour_feature
                    self._obs[self._door_idxs_start + door_idx*self.n_door_ftrs+3] = int(door.locked)

                    for i in range(n_confound_ftrs):
                        self._obs[self._door_idxs_start + door_idx*self.n_door_ftrs+4+i] = door.cnfd_ftrs[f"random_feature_{i}"]

                    if door.is_goal:
                        self._goal_object = door

                    door_idx += 1
                elif elem == 'G':
                    self._terminal_map_loc = (x, y)


        # If agent starts at a random location
        if self._actor is None:
            coords = np.argwhere(self.empty_mask)
            idx = self.rng.integers(len(coords))
            y, x = coords[idx]
            
            self._actor = Actor(loc=(x,y))
            self._obs[self._agent_idxs_start] = x
            self._obs[self._agent_idxs_start+1] = y
        
        self.agent_start_loc = self.agent_loc

    @property
    def agent_loc(self):
        return tuple(self._actor.loc)
    
    @property
    def terminal_map_loc(self):
        return self._terminal_map_loc
    
    @property
    def goal_object(self):
        return deepcopy(self._goal_object)
    
    @property
    def goal_object_loc(self):
        return tuple(self._goal_object.loc)
    
    @property
    def goal_object_id(self):
        return self._goal_object.unique_id
    
    @property
    def goal_door_locked(self):
        return self._goal_object.locked
    
    @property
    def inventory_id(self):
        if self._actor.inventory is not None:
            return self._actor.inventory.unique_id
        else:
            return None
    
    @property
    def inventory_full(self):
        return True if self._actor.inventory is not None else False
    
    @property
    def observation(self):
        if not self._one_hot:
            return self._obs.copy()

        obj_obs = self._obs[:self._door_idxs_start].reshape(self._n_objects, self.n_obj_ftrs)
        door_obs = self._obs[self._door_idxs_start:self._agent_idxs_start].reshape(self._n_doors, self.n_door_ftrs)

        obj_colour_1h = self._colour_eye[obj_obs[:, 2].astype(np.int16) - 1]
        obj_shape_1h = self._shape_eye[obj_obs[:, 3].astype(np.int16) - 1]
        door_colour_1h = self._colour_eye[door_obs[:, 2].astype(np.int16) - 1]

        if self.n_cnfd_ftrs > 0:
            obj_cnfd_1h = self._rnd_ftr_eye[obj_obs[:, 5:].astype(np.int16)].reshape(
                self._n_objects, self.n_cnfd_ftrs * RND_FTR_RNG
            )
            door_cnfd_1h = self._rnd_ftr_eye[door_obs[:, 4:].astype(np.int16)].reshape(
                self._n_doors, self.n_cnfd_ftrs * RND_FTR_RNG
            )
        else:
            obj_cnfd_1h = np.zeros((self._n_objects, 0), dtype=np.uint8)
            door_cnfd_1h = np.zeros((self._n_doors, 0), dtype=np.uint8)

        obj_encoded = np.concatenate((obj_obs[:, :2], obj_colour_1h, obj_shape_1h, obj_obs[:, 4:5], obj_cnfd_1h), axis=1)
        door_encoded = np.concatenate((door_obs[:, :2], door_colour_1h, door_obs[:, 3:4], door_cnfd_1h), axis=1)

        return np.concatenate(
            (obj_encoded.reshape(-1), door_encoded.reshape(-1), self._obs[self._agent_idxs_start:].copy())
        )
    
    def object_id_to_obs_idx(self, unique_id:int):
        obj_x = unique_id*self.n_obj_ftrs
        obj_y = unique_id*self.n_obj_ftrs + 1
        obj_colour = unique_id*self.n_obj_ftrs + 2
        obj_shape = unique_id*self.n_obj_ftrs + 3
        obj_picked_up = unique_id*self.n_obj_ftrs + 4
        return obj_x, obj_y, obj_colour, obj_shape, obj_picked_up

    def door_id_to_obs_idx(self, unique_id:int):
        obj_x = self._door_idxs_start + unique_id*self.n_door_ftrs
        obj_y = self._door_idxs_start + unique_id*self.n_door_ftrs + 1
        obj_colour = self._door_idxs_start + unique_id*self.n_door_ftrs + 2
        locked = self._door_idxs_start + unique_id*self.n_door_ftrs + 3
        return obj_x, obj_y, obj_colour, locked

    def move(self, loc:Tuple, act:str) -> bool:
        # Sets last attempted move, regardless of success
        self._actor.set_last_move(act=act)
        success = False
        x, y = loc
        
        if self._wall_mask[x,y]:
            return success
        
        if self._door_mask[x,y] and self._doors[self._door_locs.index(loc)].locked:
            return success
        
        self._actor.move(loc=loc, act=act)
        self._obs[self._agent_idxs_start:self._agent_idxs_start+2] = loc
        if self._actor.inventory is not None:
            obj_id = self._actor.inventory.unique_id
            obj_x, obj_y, _, _, _ = self.object_id_to_obs_idx(unique_id=obj_id)
            self._obs[obj_x:obj_y+1] = loc
        
        success = True
        return success
        
    def pick_up(self) -> bool:
        success = False

        for obj in self._objects:
            if obj.loc == self.agent_loc:
                success = self._actor.pick_up(obj=obj)
                if success:
                    _, _, _, _, picked_up = self.object_id_to_obs_idx(unique_id=obj.unique_id)
                    self._obs[picked_up] = 1
                break
        
        return success
        
    def drop(self) -> bool:
        success = False

        if self._actor.inventory is None:
            return success
        
        success = True
        for obj in self._objects:
            # There is another object at the position where the agent is trying to drop inventory
            if obj.loc == self.agent_loc and obj.unique_id != self._actor.inventory.unique_id:
                success = False
                break
        
        if success is False:
            return success
        
        dropped_id = self._actor.inventory.unique_id
        success = self._actor.drop()
        if success:
            # No need to change object location as it will already have been changed
            _, _, _, _, picked_up = self.object_id_to_obs_idx(unique_id=dropped_id)
            self._obs[picked_up] = 0
        
        return success

    def use_key(self) -> bool:
        success = False

        if not self._actor.inventory or not self._actor.inventory.is_key:
            return success
        
        loc_np = np.array(self.agent_loc)
        adjecent_locs = [loc_np + (0, 1), loc_np + (1, 0), loc_np + (0, -1), loc_np + (-1, 0)]
        
        for loc in adjecent_locs:
            loc = tuple(loc)
            if loc not in self._door_locs:
                continue

            door = self._doors[self._door_locs.index(loc)]
            
            if door.colour_feature == self._actor.inventory.colour_feature:
                success = True
                door.use_key()
                _, _, _, locked = self.door_id_to_obs_idx(unique_id=door.unique_id)
                self._obs[locked] = int(door.locked)
                #TODO: If multiple doors with the same colour are possible, remove break
                break
        
        return success
    
    def render_feature_grid(self, cell_size:int=60):
        """Render the grid with color fill in a vectorized manner.
        Return the upscaled color image (no text yet)."""
        rows, cols = self._wall_mask.shape

        # 1) Initialize color array: all white
        color_arr = np.full((rows, cols, 3), fill_value=(255, 255, 255), dtype=np.uint8)

        # 2) Assign black for 'W' walls
        color_arr[self._wall_mask] = (0, 0, 0)  # black

        # 4) Upscale each cell to cell_size x cell_size
        image = color_arr.repeat(cell_size, axis=0).repeat(cell_size, axis=1)
        return image
    
    def add_features(self, image, cell_size=60):
        """
        For each cell that has a feature (F0, F1, etc.):
        - Otherwise, place the two letters in the cell
        """
        small_size = cell_size
        xsmall_size = int(cell_size/2)

        agent_image = load_and_resize_png(
            path=self._actor.asset_path, 
            cell_size=cell_size,
            keep_alpha=True
        )

        goal_image = load_and_resize_png(
            path=os.path.join(ASSETS_PATH, "goal.png"),
            cell_size=cell_size,
            keep_alpha=True
        )
        
        inventory_obj = None

        for obj in self._objects:
            
            # Plotted a bit differently
            if obj.picked_up:
                inventory_obj = obj
                continue
            
            obj_img = load_and_resize_png(path=obj.asset_path, cell_size=cell_size, keep_alpha=True)
            
            x = obj.loc[0]
            y = obj.loc[1]

            y0 = x * cell_size
            x0 = y * cell_size
            overlay_with_alpha(image, obj_img, x0, y0)
            
        for door in self._doors:
            door_img = load_and_resize_png(path=door.asset_path, cell_size=cell_size, keep_alpha=True)
            x = door.loc[0]
            y = door.loc[1]
            y0 = x * cell_size
            x0 = y * cell_size
            overlay_with_alpha(image, door_img, x0, y0)

        # Plot current agent position
        y0 = self.agent_loc[0] * cell_size
        x0 = self.agent_loc[1] * cell_size
        x_offset = x0 + (cell_size - small_size) // 2
        y_offset = y0 + (cell_size - small_size) // 2
        overlay_with_alpha(image, agent_image, x_offset, y_offset)

        if inventory_obj is not None:
            y0 = inventory_obj.loc[0] * cell_size
            x0 = inventory_obj.loc[1] * cell_size
            x_offset = x0 + (cell_size - xsmall_size) // 2
            y_offset = y0 + (cell_size - xsmall_size) // 2
            inventory_img = load_and_resize_png(path=inventory_obj.asset_path, cell_size=xsmall_size, keep_alpha=True)
            overlay_with_alpha(image, inventory_img, x_offset, y_offset)
        
        # Plot goal position
        y0 = self.goal_object_loc[0] * cell_size
        x0 = self.goal_object_loc[1] * cell_size
        x_offset = x0 + (cell_size - small_size) // 2
        y_offset = y0 + (cell_size - small_size) // 2
        overlay_with_alpha(image, goal_image, x_offset, y_offset)


class Shapes(gym.Env):
    def __init__(self, objects: List[Shape], doors: List[Door], grid: List, features: Dict, store_path:str, n_confound_ftrs:int=0, max_steps:int=None, one_hot:bool=True, slip_chance:float=0, seed:int=0):
        self._store_path = store_path
        self._assets_path = ASSETS_PATH
        self._slip_chance = slip_chance
        self._one_hot = one_hot

        self._grid = deepcopy(grid)
        self._objects = deepcopy(objects)
        self._doors = deepcopy(doors)
        self._features = features
        self._n_cnfd_ftrs = n_confound_ftrs

        self.map = GameMap(grid=grid, objects=objects, doors=doors, features=features, n_confound_ftrs=self._n_cnfd_ftrs, one_hot=self._one_hot, seed=seed)
        self.observation_space = gym.spaces.MultiDiscrete([10] * self.map.observation.shape[0])

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
            3: "right",
            4: "pick_up",
            5: "drop",
            6: "use"
        }

        if max_steps is None:
            self._max_steps = 100000
        else:
            self._max_steps = max_steps
        self._steps = 0

        # UP, DOWN, LEFT, RIGHT, PICK_UP, DROP, USE
        self.action_space = gym.spaces.Discrete(7)

        # Seeding random generators for reproducibility
       
        self.action_space.seed(seed=seed)
        self.rng = np.random.default_rng(seed)
        _ = self.reset(options={"objects": objects, "doors": doors})
    
    def _init_start_location(self):
        specified_locs = np.where(self._grid == 'A')
        candidates = list(zip(*specified_locs))
        if len(candidates) == 0:
            empty_locations = np.where(self._grid == ' ')
            candidates = list(zip(*empty_locations))
        loc = self.rng.choice(candidates)
        return tuple(loc)
    
    @property
    def obs(self) -> gym.spaces.MultiDiscrete:
        return self.map.observation
    
    @property
    def goal_object(self):
        return self.map.goal_object
    
    @property
    def terminal_map_loc(self):
        return self.map.terminal_map_loc
    
    def reset(self, seed: Optional[int]=None, options: Optional[dict]={}):
        """ Reset the environment and return the initial state number
        """
        super().reset(seed=seed)
        self._steps = 0
        info = {}

        objects = options.get("objects", None)
        doors = options.get("doors", None)
        
        # We need deepcopies here because GameMap will modify these objects during the episode.
        # We don't want any info leakage between episodes.
        if objects is not None:
            self._objects = deepcopy(objects)
        else:
            objects = deepcopy(self._objects)
        
        if doors is not None:
            self._doors = deepcopy(doors)
        else:
            doors = deepcopy(self._doors)
        grid = deepcopy(self._grid)
        
        self.map = GameMap(grid=grid, objects=objects, doors=doors, features=self._features, n_confound_ftrs=self._n_cnfd_ftrs, one_hot=self._one_hot, seed=seed)
        return self.obs, info
    
    def _movement(self, action) -> bool:
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
            - 4: pick up
            - 5: drop
            - 6: use
        """
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
        
        loc_candidate = tuple(np.array(self.map.agent_loc) + self._action_to_direction[action])
        success = self.map.move(loc=loc_candidate, act=self._action_to_str[action])
        return success

    @abstractmethod
    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
            - 4: pick up
            - 5: drop
            - 6: use
        """
        if isinstance(action, torch.Tensor) or isinstance(action, np.ndarray):
            action = action.item()
        assert(action >= 0)
        assert(action <= 6)

        if action < 4:
            success = self._movement(action=action)
        elif action == 4:
            success = self.map.pick_up()
        elif action == 5:
            success = self.map.drop()
        elif action == 6:
            success = self.map.use_key()

        self._steps += 1
        info = {
            "success": success
        }
        truncated = False
        if self._max_steps is not None and self._steps >= self._max_steps:
            truncated = True
        
        # Define these in subclass
        reward = None
        is_terminal = None
        return self.obs, reward, is_terminal, truncated, info
    
    def render_frame(self) -> np.ndarray:
        image = self.map.render_feature_grid()
        self.map.add_features(image=image)
        return image

    def store_frame(self, plot_name:str='table') -> None:
        image = self.render_frame()
        output_path = os.path.join(self._store_path, f'{plot_name}.png')
        cv2.imwrite(output_path, image)


class ShapesGoto(Shapes):
    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        
        is_terminal = False
        reward = -1

        if self.map.agent_loc == self.map.goal_object_loc:
            is_terminal = True
            reward = 1
        
        return obs, reward, is_terminal, truncated, info


class ShapesSemantic(Shapes):
    def __init__(self, desireable_obj: Shape, spawned_object:Shape, doors: List[Door], grid: List, features: Dict, store_path:str, n_confound_ftrs:int=0, max_steps:int=None, one_hot:bool=True, slip_chance:float=0, seed:int=0):
        super().__init__([spawned_object], doors, grid, features, store_path, n_confound_ftrs, max_steps, one_hot, slip_chance, seed)
        self.desireable_obj = desireable_obj
        self.observation_space = gym.spaces.MultiDiscrete(
            np.concatenate((self.observation_space.nvec, np.array([2, 2], dtype=self.observation_space.nvec.dtype)))
        )
        self.picked_up_just_now = 0

    def semantic_feature_mask_0(self, obs):
        at_terminal = int(self.map.agent_loc == self.terminal_map_loc)
        at_goal_obj = int(self.map.agent_loc == self.map.goal_object_loc)
        semantic_mask = np.array([at_terminal, at_goal_obj], dtype=obs.dtype)
        return np.concatenate((obs, semantic_mask), axis=0)
    
    def semantic_feature_mask_1(self, obs):
        at_terminal = int(self.map.agent_loc == self.terminal_map_loc)
        at_goal_obj = int(self.map.agent_loc == self.map.goal_object_loc)
        bias_term = 1.0

        one_hot_colour = np.zeros(self.map.ftr_category_to_range["colour"], dtype=obs.dtype)
        colour_idx = self.map.ftr_name_to_val[self.map.goal_object.colour] - 1
        one_hot_colour[colour_idx] = 1

        one_hot_shape = np.zeros(self.map.ftr_category_to_range["shape"], dtype=obs.dtype)
        shape_idx = self.map.ftr_name_to_val[self.map.goal_object.shape] - 1
        one_hot_shape[shape_idx] = 1     

        one_hot_colour_gated = one_hot_colour * self.picked_up_just_now
        one_hot_shape_gated = one_hot_shape * self.picked_up_just_now

        semantic_mask = np.concatenate(
            (
                np.array([at_terminal, at_goal_obj, bias_term], dtype=obs.dtype),
                one_hot_colour_gated,
                one_hot_shape_gated,
            ), axis=0
        )
        return np.concatenate((obs, semantic_mask), axis=0)
    
    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        
        is_terminal = False
        reward = -1
        self.picked_up_just_now = 0

        if self.map.agent_loc == self.terminal_map_loc:
            is_terminal = True
            reward = 3    
        elif self.map.agent_loc == self.map.goal_object_loc:
            # TODO: This is very hacky. There is no goal object in this Shapes task, but if we have only one object it will be interpreted by the map
            # as the goal object so it is fine.
            if self.map.goal_object == self.desireable_obj and not self.goal_object.picked_up:
                reward = 12
                self.map.pick_up()
                self.picked_up_just_now = 1
            elif (self.map.goal_object.colour == self.desireable_obj.colour or self.map.goal_object.shape == self.desireable_obj.shape) and not self.goal_object.picked_up:
                reward = 6
                self.map.pick_up()
                self.picked_up_just_now = 1
            # elif not self.goal_object.picked_up:
            #     reward = -10
            #     self.map.pick_up()
            # self.picked_up_just_now = 1
        return obs, reward, is_terminal, truncated, info
    

class ShapesPickup(Shapes):
    def step(self, action):
        
        obs, _, _, truncated, info = super().step(action)
        
        is_terminal = False
        reward = -1

        if self.map.inventory_id == self.map.goal_object_id:
            is_terminal = True
            reward = 1
        
        return obs, reward, is_terminal, truncated, info
    

class ShapesUnlock(Shapes):
    def step(self, action):
        
        obs, _, _, truncated, info = super().step(action)
        
        is_terminal = False
        reward = -1

        if not self.map.goal_door_locked:
            is_terminal = True
            reward = 1
        
        return obs, reward, is_terminal, truncated, info


class ShapesRetrieve(Shapes):
    def step(self, action):
        
        obs, _, _, truncated, info = super().step(action)
        
        is_terminal = False
        reward = -1

        if self.map.inventory_id == self.map.goal_object_id and self.map.agent_loc == self.map.agent_start_loc:
            is_terminal = True
            reward = 1
        
        return obs, reward, is_terminal, truncated, info


if __name__ == '__main__':
    from utils import setup_artefact_paths
    from tqdm import tqdm

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path, config_name="shapes")
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)

    grid = hparams["grid"]
    features = hparams["features"]
    
    env = ShapesRetrieve(
        objects=DEFAULT_OBJECTS,
        doors=DEFAULT_DOORS,
        grid=hparams["grid"],
        features=hparams["features"],
        store_path=store_path
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
