import gymnasium as gym
import torch
import random
import numpy as np

from tetris_gymnasium.envs.tetris import Tetris

from dqn import DQN
from replay_buffer import ReplayBuffer

OBS_SIZE = 432 + 432 + 16 + 64
ACTION_SIZE = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
RB_SIZE = 100000
MINI_BATCH_SIZE = 32
EPSILON_INIT = 1
EPISILON_DECAY = 0.995
EPSILON_MIN = 0.05

class Agent:
    
    def process_state(self, state):
        board = state['board']
        atm = state['active_tetromino_mask']
        holder = state['holder']
        queue = state['queue']
        
        combined = np.append(board,atm)
        combined = np.append(combined, holder)
        combined = np.append(combined, queue)
        return combined

    def run(self, is_training=True, render=False):
        env = gym.make("tetris_gymnasium/Tetris", render_mode="human" if render else None)

        #env.reset(seed=42)

        num_states = OBS_SIZE
        num_actions = ACTION_SIZE

        rewards_per_episode = []
        epsilon_history = []

        policydqn = DQN(num_states, num_actions).to(device)

        if is_training:
            rb = ReplayBuffer(RB_SIZE)

            epsilon = EPSILON_INIT

        # Loops number of episodes
        for episode in range(1000):

            # Get state and convert to tensor
            state, _ = env.reset()
            state = self.process_state(state)
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            # Loop under one episode
            while not terminated:
                #print(env.render() + "\n")

                # Epsilon Greedy Method to get the action
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device).item()

                else:
                    action = torch.argmax(policydqn(state.unsqueeze(dim=0))).squeeze().item()

                # Step with the action
                new_state, reward, terminated, truncated, info = env.step(action)

                # Convert new_state and reward to tensors
                new_state = self.process_state(new_state)
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                # Add to episode of this reward
                episode_reward += reward

                # Add to the replay buffer if training
                if is_training:
                    rb.add((state, action, reward, new_state, terminated))

                # Set state to new state
                state = new_state
            
            # After the while loop in the episode:
            # Add to rewards list
            rewards_per_episode.append(episode_reward)

            # Update epsilon for next episode
            epsilon = max(epsilon * EPISILON_DECAY, EPSILON_MIN)

            # Add epsilon to epsilon history list
            epsilon_history.append(epsilon)

        #print("Game Over!")

    
if __name__ == "__main__":
    agent = Agent()
    agent.run(is_training=True, render=True)