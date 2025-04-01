import os
import random
import torch
import time
import numpy as np
import gym

from env_wrapper import EnvWrapper
from nets import DQN
from utils import create_run_name, visualize_result
from rb import ReplayBuffer


random.seed(0)
STORE_PATH = './tmp_deep_q_learning'


class DQNAgent(object):
    def __init__(self, state_size, action_size, gamma=0.95, batch_size=64, lr=0.0025, num_hidden=1,
                 hidden_units=32):
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.name = 'DQN'

        # We create "live" and "target" networks from the original paper.
        self.current = DQN(state_size, action_size, h=hidden_units, num_hidden=num_hidden)
        self.target = DQN(state_size, action_size, h=hidden_units, num_hidden=num_hidden)
        for p in self.target.parameters():
            p.requires_grad = False
        self.update_target_model()

        # Replay buffer (memory) initialization.
        self.rb = self.init_rb()
        self.batch_size = batch_size

        # Learning rate and optimizer used to update the "live" network in DQN.
        learning_rate = lr
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=learning_rate)

    def init_rb(self):
        # Replay buffer initialization.
        replay_buffer = ReplayBuffer(1e5, obs_dtype=np.float32, act_dtype=np.int64, default_dtype=np.float32)
        return replay_buffer

    def select_action(self, state):
        # ######## TODO: IMPLEMENT DQN ACTION SELECTION HERE ########
        # Hint: DQN uses the same greedy policy as regular Q-Learning
        action = None
        state_action_q_values = self.current.forward(state)
        action = torch.argmax(state_action_q_values).detach().item()
        return action

    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Remember (Q,S,A,R,S') as well as whether S' is terminating state.
        self.rb.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)

    def sample(self):
        states, actions, next_states, rewards, dones = self.rb.sample(self.batch_size)
        one_hot_acts = torch.squeeze(
            torch.nn.functional.one_hot(actions, num_classes=self.action_size)
        )
        return states, one_hot_acts, rewards, next_states, dones

    def backward(self):
        # ######## TODO: IMPLEMENT DQN UPDATE STEP HERE ########
        # # 1. Sample mini-batch of stored transitions from the replay buffer

        states, actions, rewards, next_states, dones = self.sample()

        # # 2. Implement the learning part of the DQN algorithm.
        # ### a. Use "states" (batch_dim x self.state_size) and "actions" (batch_dim x action_size)
        #     to get Q values (batch_dim x 1) of actions agent had played out in the environment.
        #     USE qs_selected VARIABLE TO STORE RESULT!
        qs_selected = None

        qs = self.current(states)
        qs_selected = torch.sum(qs * actions, dim=1)

        # ### b. Use "next_states" (batch_dim x self.state_size) to calculate the target value
        #     Q values obtained in a. should be regressed to.
        #     Hint 1: Target network plays important part here!
        #     Hint 2: Pay attention to PyTorch gradient accumulation!
        #     Hint 3: You can use "dones" (batch_dim x 1) to check whether "next_states" are
        #     terminating!
        #     USE qs_target VARIABLE TO STORE RESULT!
        qs_target = None

        with torch.no_grad():
            qs_nxt = self.target(next_states)
            qs_t_opt = torch.max(qs_nxt, dim=1)[0]
            qs_target = torch.squeeze(rewards) + (1. - torch.squeeze(dones)) * self.gamma * qs_t_opt

        # Code below updates the "live" network self.current using the variables you have
        # calculated in TODO section above: qs_selected and qs_target. DO NOT MODIFY THIS CODE.

        # We calculate the absolute difference between current and target values q values,
        # which is useful info for debugging.
        with torch.no_grad():
            td_error = torch.abs(qs_target - qs_selected)

        # We update the "live" network, self.current. First we zero out the optimizer gradients
        # and then we apply the update step using qs_selected and qs_target.
        self.optimizer.zero_grad()
        loss = (torch.nn.functional.mse_loss(qs_selected, qs_target)).mean()
        loss.backward()
        self.optimizer.step()
        return torch.mean(td_error).item()


# ### HYPER PARAMETERS ##########
# ### EPISODE ###################
# Number of episodes to train for
EPISODES = 250
# Number of steps per episode
STEPS = 200

# ### TRAINING ##################
# Start exploration rate
# for eps greedy policy
EPSILON_START = 1
# End exploration rate
EPSILON_END = 0.001
# Exploration rate decay
# per EPISODE
EPSILON_DECAY = 0.985
# Discount rate
GAMMA = 0.95
# DQN Target network update
# frequency in EPISODES
TARGET_FREQ = 4
# DQN live network update
# frequency in STEPS
UPDATE_FREQ = 1
# Learning rate
LR = 0.00015
# Batch size
BATCH_SIZE = 256

# ### NETWORK ARCHITECTURE ######
# Number of hidden layers
# of H units
NUM_H = 2
# Number of units in hidden
# layers
H = 64
# ### HYPER PARAMETERS END #######
run_name = create_run_name(
        alg='DQN',
        env='stick',
        num_layers=NUM_H,
        hidden_dim=H,
        eps_start=EPSILON_START,
        eps_end=EPSILON_END,
        decay=EPSILON_DECAY,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_ep=EPISODES,
        num_step=STEPS,
        updt_freq=UPDATE_FREQ,
        sw_freq=TARGET_FREQ
    )


def main():
    env = EnvWrapper(gym_env=gym.make('CartPole-v0'), steps=STEPS)
    agent = DQNAgent(
        state_size=env.state_size(),
        action_size=env.action_size(),
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_hidden=NUM_H,
        hidden_units=H
    )

    epsilon = EPSILON_START
    results = []
    td_errors = []
    start = time.time()
    random.seed(0)

    for episode in range(EPISODES):
        # Start game/episode
        state = env.reset()
        cum_rew = 0

        if episode > 10 and episode % TARGET_FREQ == 0:
            agent.update_target_model()

        # Loop inside one game episode
        for t in range(STEPS):
            # Display the game. Comment bellow line in order to get faster training.
            # env.render()
            if random.random() <= epsilon:
                action = env.env.action_space.sample()
            else:
                action = agent.select_action(state=torch.from_numpy(state).detach())

            next_state, reward, done = env.step(action)
            cum_rew += reward

            agent.remember(state=state, action=action, reward=reward, next_state=next_state, done=float(done))

            if episode > 10 and (episode + t) % UPDATE_FREQ == 0:
                td_error = agent.backward()
                td_errors.append(td_error)

            if done or (t == STEPS - 1):
                if episode > 10:
                    print("EPISODE: {0: <4}/{1: >4} | EXPLORE RATE: {2: <7.4f} | SCORE: {3: <7.1f}"
                          " | TD ERROR: {4: <5.2f} ".format(episode + 1, EPISODES, epsilon, cum_rew, td_error))
                else:
                    print("EPISODE: {0: <4}/{1: >4} | EXPLORE RATE: {2: <7.4f} | SCORE: {3: <7.1f}"
                          " | WARMUP - NO TD ERROR".format(episode + 1, EPISODES, epsilon, cum_rew))
                results.append(cum_rew)
                break

            state = next_state

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

    end = time.time()
    total_time = end - start

    print()
    print("TOTAL EPISODES: {0: <4} | TOTAL UPDATE STEPS: {1: <7} | TOTAL TIME [s]: {2: <7.2f}"
          .format(EPISODES, len(td_errors), total_time))
    print("EP PER SECOND: {0: >10.6f}".format(total_time / EPISODES))
    print("STEP PER SECOND: {0: >8.6f}".format(total_time / len(td_errors)))

    fig = visualize_result(
        returns=results,
        td_errors=td_errors,
        policy_errors=None
    )
    fig.show()
    fig.savefig(os.path.join(STORE_PATH, run_name + '.png'), dpi=400)


if __name__ == '__main__':
    main()
