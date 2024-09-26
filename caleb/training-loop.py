import random
import torch
import torch.nn.functional as F

def train(model, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    # Sample a batch of transitions
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute Q-values for current states
    q_values = model(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values for next states
    next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Compute the loss
    loss = F.mse_loss(q_values, target_q_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()