from tetris_gymnasium.envs import Tetris
import gymnasium as gym
from nes_py.wrappers import JoypadSpace

env = gym.make("tetris_gymnasium/Tetris")


print(env.action_space)