import cv2
import gymnasium as gym
import torch

from tetris_gymnasium.envs.tetris import Tetris
from train import DQN, flatten, device

PATH = 'DQN-seed200-16000CumEps.pth'


import torch
import gymnasium as gym
import numpy as np

# Define your DQN class, flatten function, and other required components here...



# Set up the Tetris environment
env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset(seed=200)
terminated = False

state = flatten(state)
n_observations = len(state)

# Load the trained model
policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("DQN-seed200-64000CumEps.pth"))
policy_net.eval()




for i in range(10):
    
    state, info = env.reset(seed=200)
    state = flatten(state)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    score = 0
    while not done:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)  # Select action using the trained model
        observation, reward, terminated, truncated, _ = env.step(action.item())
        score += reward
        print(f'reward {reward}')
        env.render()
        key = cv2.waitKey(100)
        # Update state
        state = flatten(observation)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        done = terminated or truncated
    print(f'iteration {i} score {score}')

env.close()
