from typing import List, Optional, Tuple
import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm


class Gridworld(gym.Env):
    def __init__(self, grid: List, store_path:str, slip_chance:float=0, step_r: float=-1, goal_r: float=10, goal_pos: Tuple=(4, 5), start_pos: Tuple=None, max_steps:int=None, seed:int=None):
        self._grid = np.array(grid)
        self._walls = np.equal(self._grid, 'W')
        self._agent_location = start_pos
        self._target_location = goal_pos
        self._action_to_direction = {
            0: np.array([-1, 0]), # up
            1: np.array([1, 0]),  # down
            2: np.array([0, -1]), # left
            3: np.array([0, 1]),  # right
        }
        self._slip_chance = slip_chance
        self._max_steps = max_steps
        self._steps = 0
        self.action_space = gym.spaces.Discrete(4)

        # Seeding random generators for reproducibility
        if seed is not None:
            self.action_space.seed(seed=seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.observation_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([self._grid.shape[0], self._grid.shape[1]]), shape=(2,), dtype=int)
        
        # Random start option
        self._random_start = start_pos == None
        if not self._random_start:
            self.start_pos = start_pos
        
        self.reward = {' ': step_r, 'G': goal_r}
        self.store_path = store_path
        
        _ = self.reset()

    @property 
    def wall_mask(self) -> np.ndarray:
        return self._walls
    
    @property
    def n_states(self) -> np.ndarray:
        return np.prod(self.observation_space.high)
    
    @property
    def grid_shape(self) -> np.ndarray:
        return self._grid.shape
    
    def _get_obs(self) -> np.ndarray:
        return np.array(self._agent_location, dtype=int)
    
    def _pick_random_start(self):
        x = self.rng.integers(1, self.grid_shape[0])
        y = self.rng.integers(1, self.grid_shape[1])
        while self._grid[x, y] == 'W' or (x, y) == self._target_location:
            x = self.rng.integers(1, self.grid_shape[0])
            y = self.rng.integers(1, self.grid_shape[1])
        return x, y
    
    def obs_to_ids(self, observations):
        """
        Generates unique indices for a batch of observations or a single observation,
        supporting both NumPy arrays and PyTorch tensors.
        Args:
            observations (np.ndarray or torch.Tensor): Environment observation(s).
                - Shape (2,) for a single observation.
                - Shape (N, 2) for a batch of observations.
        Returns:
            np.ndarray or torch.Tensor: Array of unique IDs (shape (N,)).
        """
        is_tensor = torch.is_tensor(observations)  # Check if input is a torch tensor
        
        # Ensure observations are at least (1,2) shape
        if observations.ndim == 1:
            observations = observations.unsqueeze(0) if is_tensor else np.expand_dims(observations, axis=0)

        obs_ids = observations[:, 0] * self.observation_space.high[1] + observations[:, 1]

        # Convert result to int type
        return obs_ids.int() if is_tensor else obs_ids.astype(int)
    
    def ids_to_obs(self, obs_ids):
        """
        Converts unique observation IDs back to observations, supporting both NumPy arrays and PyTorch tensors.
        Args:
            obs_ids (np.ndarray, torch.Tensor, or int): Unique observation ID(s).
                - Shape (N,) for a batch of IDs.
                - Shape (1,) or a plain int for a single ID.
        Returns:
            np.ndarray or torch.Tensor: Array of observations (shape (N,2)), or (2,) for a single observation.
        """
        is_tensor = torch.is_tensor(obs_ids)  # Check if input is a PyTorch tensor

        # Convert scalar or (1,) shape to batch
        if np.isscalar(obs_ids) or (is_tensor and obs_ids.numel() == 1) or (not is_tensor and isinstance(obs_ids, np.ndarray) and obs_ids.shape == (1,)):
            obs_ids = torch.tensor([obs_ids]) if is_tensor else np.array([obs_ids])

        # Compute x and y coordinates
        y = obs_ids % self.observation_space.high[1]
        x = obs_ids // self.observation_space.high[1]

        # Stack results
        observations = torch.stack((x, y), dim=1) if is_tensor else np.stack((x, y), axis=1)

        # Return single observation as (2,) instead of (1,2)
        return observations[0] if observations.shape[0] == 1 else observations
    
    def acts_to_ids(self, actions):
        """
        Converts actions to their corresponding IDs, handling both single and batch inputs.
        Args:
            actions (np.ndarray, torch.Tensor, or int): Action(s).
                - A single action as an int or an array of shape (1,).
                - A batch of actions as an array of shape (N,).
        Returns:
            np.ndarray, torch.Tensor, or int: Action ID(s).
                - Returns an int if the input was a single action.
                - Returns an array of shape (N,) if the input was a batch.
        """
        is_tensor = torch.is_tensor(actions)  # Check if input is a PyTorch tensor

        # Convert scalar or (1,) shape to batch
        if np.isscalar(actions) or (is_tensor and actions.numel() == 1) or (not is_tensor and isinstance(actions, np.ndarray) and actions.shape == (1,)):
            actions = torch.tensor([actions]) if is_tensor else np.array([actions])

        # Convert to int type
        action_ids = actions.int() if is_tensor else actions.astype(int)

        # Return single action as a scalar
        return action_ids[0] if action_ids.shape[0] == 1 else action_ids
    
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
        assert self._grid[self._agent_location] != 'W'
        return self._get_obs(), info

    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
        """
        assert(action >= 0)
        assert(action <= 3)
        
        # If the agent slips
        if self.rng.random() < self._slip_chance:
            if action == 0:
                action = self.rng.choice([2, 3])
            elif action == 1:
                action = self.rng.choice([3, 2])
            elif action == 2:
                action = self.rng.choice([1, 0])
            else:
                action = self.rng.choice([0, 1])

        agent_location = tuple(self._get_obs() + self._action_to_direction[action])
        if self._grid[agent_location] != 'W':
            self._agent_location = agent_location
        assert self._grid[self._agent_location] != 'W'

        self._steps += 1
        is_terminal = self._agent_location == self._target_location
        info = ''

        truncated = False
        if self._max_steps is not None and self._steps >= self._max_steps:
            truncated = True
        return self._get_obs(), self.reward[self._grid[self._agent_location]], is_terminal, truncated, info

    def plot_values(self, table: np.ndarray, plot_name='table', colormap: str = 'viridis', add_walls: bool=True):
        """
        Generates a heatmap from a 2D numpy array and saves it to the specified file path.

        Parameters:
        - table (np.ndarray): The 2D numpy array to visualize as a heatmap.
        - colormap (str): The name of the colormap to use (default is 'viridis').

        Returns:
        None
        """

        if table.ndim != 2:
            raise ValueError("Input array must be two-dimensional.")

        if add_walls is True:
            table = np.where(self.wall_mask, np.nan, table)
            # Create the colormap with black for NaN values
            colormap = cm.get_cmap(colormap).copy()
            colormap.set_bad(color='black')  # Set NaN (or masked values) to black

        # Create the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(table, cmap=colormap, aspect='auto', origin='upper')
        plt.colorbar(label='Value')  # Display color bar for mapping values to colors
        plt.title(plot_name)
        
        # Save the heatmap to the specified file path
        plt.savefig(os.path.join(self.store_path, f'{plot_name}.png'), format='png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    from utils import setup_artefact_paths

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path)
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
    env = Gridworld(
        grid=hparams['grid'], 
        start_pos=tuple(hparams['start_pos']),
        store_path=store_path, 
        max_steps=hparams['max_steps'],
        seed=hparams['seed']
    )
    visitations = np.zeros(env.observation_space.high)

    for episode in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset()
        done = False

        while not done:
            visitations[tuple(obs)] += 1
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
    
    env.plot_values(table=visitations, plot_name='visitations')
