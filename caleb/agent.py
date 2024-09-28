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
        self.memory.append(transition)

    def choose_action(self, observation):
        # Epsilon-greedy action selection
        u = np.random.rand()  # Generates a random float in [0, 1)
        if u < self.epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5])  # Random action for exploration
        else:
            # Choose the best action based on Q-values for the given observation
            with torch.no_grad():
                q_values = self.model(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
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

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values for current states
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values for next states
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss
        loss = F.mse_loss(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Usage example
# Assuming you have a model and optimizer defined
# agent = DQNAgent(model, optimizer, gamma=0.99, batch_size=32)

# In your training loop:
# agent.remember((state, action, reward, next_state, done))
# action = agent.choose_action(state)
# agent.train()
