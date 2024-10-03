from tetris_gymnasium.envs import Tetris
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch

env = gym.make("tetris_gymnasium/Tetris")

observation, info = env.reset()
board = observation['board']
print(type(board))

print(type(torch.as_tensor(board)))

'''
print("Start State1: \n", str(observation['board'])[1:-1])
print("Info1: ", info)

observation, info = env.reset()
print("Start State2: \n", str(observation['board'])[1:-1])
print("Info2: ", info)
'''

'''
a = env.action_space.sample()
print(a)
#print(env.observation_space.sample())
observation, reward, terminated, truncated, info = env.step(a)
print("Observation: ", observation)
print("Reward: ", reward)
print("Terminated: ", terminated)
print("Truncated: ", truncated)
print("Info: ", info)
'''