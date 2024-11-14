import torch
import numpy as np
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
state,_ = env.reset()

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

t = torch.tensor(1.02)
print(t)
print(t.item())


EPSILON = 2

def change():
    EPSILON = 4


change()

print(EPSILON)
