from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers import grouped
import gymnasium as gym
import numpy as np
import torch
import cv2

env = gym.make("tetris_gymnasium/Tetris", render_mode = "human")

start_state, info = env.reset()
action = env.action_space.sample()  # You'll need a proper policy here
env.render()
cv2.waitKey(1000)
new_state, reward, terminated, truncated, info = env.step(action)
img = env.render()
cv2.waitKey(1000)

print(type(env))

print(type(reward))


