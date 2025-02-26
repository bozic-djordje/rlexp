import os
from typing import Any
import gymnasium as gym
import numpy as np
from tqdm import tqdm

class Taxicab(gym.Env):
    def __init__(self):
        self.base = gym.make("Taxi-v3", render_mode="human")
    
    @property
    def action_space(self):
        return self.base.action_space

    @property
    def observation_space(self):
        return self.base.observation_space
    
    def reset(self):
        obs, info = self.base.reset()
        return obs, info 
    
    def step(self, action: Any):
        next_obs = self.base.step(action=action)
        return next_obs


if __name__ == '__main__':
    from utils import setup_artefact_paths

    script_path = os.path.abspath(__file__)
    store_path, yaml_path = setup_artefact_paths(script_path=script_path)
    
    import yaml
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    
    env = Taxicab()
    
    for episode in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated

