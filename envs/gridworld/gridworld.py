from typing import List, Optional, Tuple
import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm


# Gridworld ' ' is a regular cell, 'G' is the goal, and 'W' is the wall.
GRID = [
    ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', 'W', 'W', 'W', 'W', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', 'W', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', 'W', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', 'W', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', 'W', 'W', 'W', 'W', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
]
ACTIONS = ['^', 'v', '<', '>']


class Gridworld(gym.Env):
    # TODO: Make stochastic
    def __init__(self, store_path, grid: List=GRID, step_r: float=-1, goal_r: float=10, goal_pos: Tuple=(4, 5), start_pos: Tuple=(4, 7), max_steps:int=None):
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
        self._max_steps = max_steps
        self._steps = 0

        self.observation_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([self._grid.shape[0], self._grid.shape[1]]), shape=(2,), dtype=int)
        self.action_space = gym.spaces.Discrete(4)
        
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
    
    def _get_obs(self) -> np.ndarray:
        return np.array(self._agent_location, dtype=int)
    
    def obs_to_id(self, observation: np.ndarray) -> int:
        """Generates unique index for each observation. Useful for tabular agent methods.
        Args:
            observation (np.ndarray): Environment observation

        Returns:
            int: Unique id of the observation.
        """
        return observation[0]*self.observation_space.high[1] + observation[1]
    
    def act_to_id(self, action: int) -> int:
        return action
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """ Reset the environment and return the initial state number
        """
        super().reset(seed=seed)
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
    
    n_episodes = 10
    max_steps = 100
    
    env = Gridworld(store_path=os.path.dirname(os.path.abspath(__file__)), max_steps=max_steps)
    visitations = np.zeros(env.observation_space.high)

    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False

        while not done:
            visitations[tuple(obs)] += 1
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
    
    env.plot_values(table=visitations, plot_name='visitations')
