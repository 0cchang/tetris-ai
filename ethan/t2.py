import torch
import numpy
import cv2

import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    step = 0
    terminated = False
    while not terminated:
        env.render()
        cv2.waitKey(100)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation['board'])
    print("Game Over!")