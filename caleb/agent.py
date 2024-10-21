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
        self.memory = deque(maxlen=memory_size) # (s, a, r, s', done)

        # Initialize epsilon values
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def remember(self, transition):
        """ Store a transition in memory. """
        self.memory.append(transition)

    def choose_action(self, observation_tensor):
        """ Epsilon-greedy action selection. """
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])  # Random action for exploration
        else:
            # Choose the best action based on Q-values for the given observation
            with torch.no_grad():
                q_values = self.model(observation_tensor)
                action = torch.argmax(q_values).item()  # Best action based on Q-values
        
        # Decay epsilon after choosing an action
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        return action

    def train(self, observation_tensor, target):
        """Train the agent using the chosen action and the corresponding target."""
        # Set the model to evaluation mode
        self.model.eval()

        # Obtain the Q-values for the current observation
        with torch.no_grad():
            q_values = self.model(observation_tensor)  # Assuming this returns a tensor of shape (num_actions,)

        # Choose an action using epsilon-greedy strategy
        action = self.choose_action(observation_tensor)  # action is a single integer

        # Gather the Q-value for the chosen action
        selected_q_value = q_values[action]  # Direct indexing to get the Q-value for the chosen action

        # Calculate the loss (for example, using Mean Squared Error)
        # Assuming target is a tensor with the same shape as selected_q_value
        loss = (selected_q_value - target).pow(2).mean()

        # Backpropagation
        self.model.train()  # Switch back to training mode
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update the model's weights

        # Optionally print or return loss for monitoring
        print(f"Action: {action}, Selected Q-value: {selected_q_value.item()}, Loss: {loss.item()}")


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
