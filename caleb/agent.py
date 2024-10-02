import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, model, optimizer, gamma, batch_size, memory_size=10000, epsilon_init=1.0, epsilon_decay=0.99, epsilon_end=0.01):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Initialize epsilon values
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def remember(self, transition):
        """ Store a transition in memory. """
        self.memory.append(transition)

    def choose_action(self, observation):
        """ Epsilon-greedy action selection. """
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])  # Random action for exploration
        else:
            # Choose the best action based on Q-values for the given observation
            with torch.no_grad():
                observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                q_values = self.model(observation_tensor)
                action = torch.argmax(q_values).item()  # Best action based on Q-values

        # Decay epsilon after choosing an action
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        return action

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of transitions
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of states to tensors
        states = torch.stack([torch.tensor(state, dtype=torch.float32) for state in states])
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)  # Unsqueeze actions to have shape (batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack([torch.tensor(next_state, dtype=torch.float32) for next_state in next_states])
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values for current states
        q_values = self.model(states)

        # Use gather to get the Q-values for the actions taken
        q_values = q_values.gather(1, actions)  # Now actions has the shape (batch_size, 1)

        # Compute target Q-values for next states
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss
        loss = F.mse_loss(q_values.squeeze(1), target_q_values)  # Squeeze to match dimensions for loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def reset_epsilon(self):
        """ Reset epsilon to its initial value. """
        self.epsilon = self.epsilon_init

# Usage example
# Assuming you have a model and optimizer defined
# agent = DQNAgent(model, optimizer, gamma=0.99, batch_size=32)

# In your training loop:
# agent.remember((state, action, reward, next_state, done))
# action = agent.choose_action(state)
# agent.train()
