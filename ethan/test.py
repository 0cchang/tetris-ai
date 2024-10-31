from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers import grouped
import gymnasium as gym
import numpy as np
import torch
import cv2

env = gym.make("tetris_gymnasium/Tetris", render_mode = "human")

observation, info = env.reset()
action = env.action_space.sample()  # You'll need a proper policy here




print(observation['board'].squeeze())

