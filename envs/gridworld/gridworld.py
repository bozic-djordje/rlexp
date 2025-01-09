from typing import List, Tuple
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
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


class Gridworld(object):
    # TODO: Make stochastic
    def __init__(self, store_path, grid: List=GRID, step_r: float=-1, goal_r: float=10, start_pos: Tuple=(4, 7), goal_pos: Tuple=(4, 5)):
        
        self.start_pos = start_pos
        self.grid = np.array(grid)
        self.grid[goal_pos] = 'G'
        self.goal_pos = goal_pos

        self.actions = ACTIONS
        self.reward = {' ': step_r, 'G': goal_r}

        self._x = self._y = None
        self.store_path = store_path
        _ = self.reset()
    
    @property
    def current_state(self):
        return self._x, self._y
    
    @property
    def action_space(self):
        return 4,

    @property
    def obs_space(self):
        return self.grid.shape
    
    @property 
    def wall_mask(self) -> np.ndarray:
        walls = np.equal(self.grid, 'W')
        return walls

    def reset(self):
        """ Reset the environment and return the initial state number
        """
        self._x = self.start_pos[0]
        self._y = self.start_pos[1]
        assert self.grid[self.current_state] != 'W'
        return self.current_state

    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:
            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
        """
        assert(action >= 0)
        assert(action <= 3)

        x = self._x
        y = self._y

        # Go up
        if action == 0:
            x -= 1
        # Go down
        elif action == 1:
            x += 1
        # Go left
        elif action == 2:
            y -= 1
        # Go right
        elif action == 3:
            y += 1
        
        if self.grid[x, y] != 'W':
            self._x = x
            self._y = y

        # Return the current state, a reward and whether the episode terminates
        cell = self.grid[self._x, self._y]
        assert cell != 'W'
        is_terminal = cell == 'G'

        return self.current_state, self.reward[cell], is_terminal

    def random_action(self):
        return np.random.choice(self.action_space[0])

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

    def print_qtable_stats(self, qtable):
        # zeros = np.zeros(size=self.action_space(), dtype=np.float16)
        # print('ENV')
        # for y in range(self.env_space()[1]):
        #     print(self.grid[y])
        # print('QACTIONS')
        # print(self.actions)
        # print('QVALUES')
        # for x in range(self.env_space()[1]):
        #     for y in range(self.env_space()[0]):
        #         print("x,y:{0},{1}, f:{2}, qvalues:{3}".format(x, y, self.grid[y][x], qtable[y, x]))
        #     print()
        # print('GREEDY POLICY')
        # for y in range(self.env_space()[0]):
        #     for x in range(self.env_space()[1]):
        #         if torch.equal(qtable[y,x], zeros):
        #             print('?', end='  ')
        #         else:
        #             print(self.actions[torch.argmax(qtable[y, x])], end='  ')
        #     print()
        pass


if __name__ == '__main__':
    
    n_episodes = 10
    max_steps = 100
    
    env = Gridworld(store_path=os.path.join(os.getcwd(), 'envs', 'gridworld'))
    visitations = np.zeros(env.obs_space)

    for episode in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        n_steps = 0

        while not done:
            visitations[obs] += 1
            action = env.random_action()
            next_obs, reward, terminated = env.step(action)
            
            obs = next_obs
            n_steps += 1
            done = terminated or n_steps > max_steps
    
    env.plot_values(table=visitations, plot_name='visitations')

