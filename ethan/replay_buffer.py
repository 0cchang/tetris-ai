from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, new_states, terminations = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.stack(rewards)
        new_states = torch.stack(new_states)
        terminations = torch.tensor(terminations).float()
        return states, actions, rewards, new_states, terminations
    
    def __len__(self):
        return len(self.buffer)
    
    def __str__(self):
        return self.buffer.__str__()
    