import gymnasium as gym
import numpy as np
import cv2
from tetris_gymnasium.envs.tetris import Tetris
from agent import DQNAgent
from training_model import DQN
import torch.optim as optim

# Create an optimizer for the DQN model
 # Learning rate of 0.001

# Action space: [x change, rotation]
# x change = -4 (move left) to 5 (move right)
# rotation = 0 (no rotation), 1, 2, 3 (CCW rotations) or negative values (CW rotations)


def train(env, num_episodes=1000):
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    gamma = 0.99  # Discount factor
    batch_size = 32  # Batch size for training
    agent = DQNAgent(model, optimizer, gamma, batch_size, memory_size=10000)

    for episode in range(num_episodes):
        observation, info = env.reset(seed=42)
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            # Choose an action based on the current observation
            action = agent.choose_action(observation)

            # Execute the action in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Accumulate reward
            total_reward += reward

            # Render environment (optional for visualization)
            env.render()

            # Optional: Add key wait to see movement step-by-step (for testing purposes)
            key = cv2.waitKey(1)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    print("Training Complete!")

if __name__ == "__main__":
    # Create the Tetris environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    
    # Start training the agent
    train(env, num_episodes=100)

    # Close the environment after training is done
    env.close()
