# import necessary libs
import torch
import torch.nn as nn
import torch.optim as optim
import numpy

from tetris_gymnasium.envs import Tetris
import gymnasium as gym

from dqn import *
from replay_buffer import *

# Create necessary constants
BOARD_SIZE = 20*10
CURR_PIECE_SIZE = 2*4
NEXT_PIECE_SIZE = 2*4
ACTION_SPACE_SIZE = 8

INPUT_DIM = BOARD_SIZE + CURR_PIECE_SIZE + NEXT_PIECE_SIZE
OUPUT_DIM = ACTION_SPACE_SIZE

BUFFER_CAPACITY = 100

# Create DQN model and initialize parameters
model = DQN(input_dim=INPUT_DIM, output_dim=OUPUT_DIM)

# Create environment
# - specify simplified action space
env = gym.make("tetris_gymnasium/Tetris")

# Instantiate replay buffer
rb = ReplayBuffer(BUFFER_CAPACITY)

# Make first observation about game state
init_obs, _ = env.reset()


# Loop
for i in range(10):

    # Randomly select action to take
    # random_action = env.action_space.sample()
    state = torch.rand(1, INPUT_DIM)  # Example random state
    q_values = model(state)  # Forward pass to get Q-values

    random_action = torch.argmax(q_values).item()

    print("Action: ", random_action)
    # Take this action in the game
    new_obs, reward, terminated, truncated, info = env.step(random_action)

    # Get new game state with reward/loss and store in replay buffer
    rb.add(init_obs, random_action, reward, new_obs, terminated)

    # Set this new game state as the starting state
    init_obs = new_obs

    print(reward)

'''
For however many episodes we want
    Enter the starting state
    Select Best action (epsilon greedy method)
    Perform action
    Get new game state with reward/loss and store in replay buffer
    state = new state
'''
    