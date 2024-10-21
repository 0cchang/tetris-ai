from tetris_gymnasium.envs import Tetris
import gymnasium as gym
import numpy as np
import torch
import cv2

env = gym.make("tetris_gymnasium/Tetris", render_mode = "human")

observation, info = env.reset()
board = observation['board']

action = env.action_space.sample()  # You'll need a proper policy here

env.step(action)

env.render()
#cv2.waitKey(10000)
cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE)
cv2.waitKey(1000)


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