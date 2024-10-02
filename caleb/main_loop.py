import torch
import torch.optim as optim
from agent import DQNAgent
from training_model import DQN
import gymnasium as gym
import cv2
from tetris_gymnasium.envs.tetris import Tetris
import numpy as np

def process_observation_gym(observation):
    active_tetromino_mask = observation['active_tetromino_mask'].flatten()
    board = observation['board'][1:-4, 4:-4].flatten()

    holder = observation['holder'].flatten()
    queue = observation['queue'][:, :10].flatten()
    processed_observation = np.concatenate([active_tetromino_mask, board, holder, queue])
    
    return processed_observation

def train(env, num_episodes=1000):
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DQN model and move to GPU if available
    model = DQN(678, 8).to(device)  # Adjust input/output sizes as needed
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    gamma = 0.99
    batch_size = 32
    epsilon = 1.0  # Starting epsilon for exploration
    epsilon_decay = 0.995
    epsilon_min = 0.01

    agent = DQNAgent(model, optimizer, gamma, batch_size, memory_size=10000)

    for episode in range(num_episodes):
        observation, info = env.reset(seed=42)
        observation = process_observation_gym(observation)  # Process the initial observation
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            # Convert observation to a tensor and move to GPU
            observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

            # Choose an action based on the current observation
            action = agent.choose_action(observation_tensor)

            # Execute the action in the environment
            next_observation, reward, terminated, truncated, info = env.step(action)

            # Process the next observation
            next_observation = process_observation_gym(next_observation)

            # Move data to GPU for storing in memory
            next_observation_tensor = torch.tensor(next_observation, dtype=torch.float32).unsqueeze(0).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
            
            # Store the transition (s, a, r, s', done)
            agent.remember((observation_tensor, action, reward_tensor, next_observation_tensor, terminated))

            # Train the agent
            agent.train()

            # Update the current observation
            observation = next_observation

            total_reward += reward

            # Render the environment
            env.render()
            key = cv2.waitKey(1)

        # Epsilon decay after each episode
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    print("Training Complete!")

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    train(env, num_episodes=100)
    env.close()
