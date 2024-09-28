import random
import torch
import torch.nn.functional as F
from collections import deque

class DQNAgent:
    def __init__(self, model, optimizer, gamma, batch_size, memory_size=10000):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

    def remember(self, transition):
        self.memory.append(transition)

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
# agent.train()
