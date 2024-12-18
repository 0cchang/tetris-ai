import torch
import torch.optim as optim
from agent import DQNAgent
from training_model import DQN
import gymnasium as gym
import cv2
from tetris_gymnasium.envs.tetris import Tetris
import numpy as np
'''
def process_observation_gym(observation):
    active_tetromino_mask = observation['active_tetromino_mask'].flatten()
    board = observation['board'][1:-4, 4:-4].flatten()

    holder = observation['holder'].flatten()
    queue = observation['queue'][:, :10].flatten()
    processed_observation = np.concatenate([active_tetromino_mask, board, holder, queue])
    
    return processed_observation

print(gym.envs.registry.keys())

env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
observation_space = env.observation_space
print(observation_space)
action_space = env.action_space
print(action_space)

observation, info = env.reset(seed=42)

print(len(process_observation_gym(observation)))
env.close()
'''
print(torch.version.cuda)
print("cuda:",torch.cuda.is_available())

