from abc import abstractmethod
from gymnasium import Env
from tqdm import tqdm
from typing import Dict

# TODO: Use this class to define some Agent properties, now useful for type hinting
class Agent:
    
    @abstractmethod
    def store_transition():
        pass

    @abstractmethod
    def update():
        pass

def train_loop(env: Env, agent: Agent, hparams: Dict) -> None:
    for _ in tqdm(range(hparams['n_episodes'])):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(
                obs=obs, 
                next_obs=next_obs,
                action=action, 
                reward=reward, 
                terminated=terminated, 
                truncated=truncated
            )
            if hparams['algo_type'] == 'off_policy':
                agent.update(batch_size=hparams['batch_size'])
            
            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
        
        # On-policy psi update
        if hparams['algo_type'] == 'on_policy':
            agent.update()