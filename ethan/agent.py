import gymnasium as gym
import torch
import random
import numpy as np
import cv2

from tetris_gymnasium.envs.tetris import Tetris

from dqn import DQN
from replay_buffer import ReplayBuffer

OBS_SIZE = 432 + 432 + 16 + 64
ACTION_SIZE = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
RB_SIZE = 100000
MINI_BATCH_SIZE = 32
EPISILON_DECAY = 0.9995
EPSILON_MIN = 0.05
NETWORK_SYNC_RATE = 10
DISCOUNT_FACTOR = 0.99
LEARNING_RATE_A = 0.0001

class Agent:
    
    '''
    3 variables:
        - self.best_reward_file: best reward file name
        - self.best_reward: best reward from the file
        - self.model: the tmp file that the model is saved and loaded
    '''
    def __init__(self, mode=1, starting_epsilon=1, num_episodes=100):
        self.mode = mode
        if mode == 1:
            self.best_reward_file = "ethan/sbr.txt"
            self.model = "ethan/siterewardtrain.tmp"
        elif mode == 2:
            self.best_reward_file = "ethan/dbr.txt"
            self.model = "ethan/downloadrewardtrain.tmp"
        elif mode == 3:
            self.best_reward_file = "ethan/sbr_nb.txt"
            self.model = "ethan/nobumpytrain.tmp"

        best_reward_file = open(self.best_reward_file)
        self.best_reward = torch.tensor(float(best_reward_file.read()))
        best_reward_file.close()

        # Changing Hyperparameters
        self.epsilon_init = starting_epsilon
        self.episodes = num_episodes

    def process_state(self, obs):
        # Normalize each component
        board = torch.tensor(obs['board'], dtype=torch.float32, device=device) / 9.0
        active_tetromino_mask = torch.tensor(obs['active_tetromino_mask'], dtype=torch.float32, device=device)
        holder = torch.tensor(obs['holder'], dtype=torch.float32, device=device) / 9.0
        queue = torch.tensor(obs['queue'], dtype=torch.float32, device=device) / 9.0
        
        # Flatten each component
        board_flat = board.view(-1)
        active_tetromino_mask_flat = active_tetromino_mask.view(-1)
        holder_flat = holder.view(-1)
        queue_flat = queue.view(-1)
        
        # Concatenate into a single tensor
        state_tensor = torch.cat([board_flat, active_tetromino_mask_flat, holder_flat, queue_flat])
        
        return state_tensor

    def bump(self, board):
        # Gets the heights of each col
        heights = np.array([0]*10)
        h_done = np.array([False]*10)
        for row in range(19, -1, -1):
            for col in range(4,14):
                if h_done[col-4]:
                    continue
                if board[row][col] != 0:
                    heights[col-4] += 1
                else:
                    h_done[col-4] = True

        return np.var(heights)
    
    def run(self, is_training=True, render=False, bumpy=True):
        print("Initializing environment...")
        env = gym.make("tetris_gymnasium/Tetris", render_mode="human" if render else None)

        num_states = OBS_SIZE
        num_actions = ACTION_SIZE

        rewards_per_episode = []

        print("Creating policy DQN")
        policydqn = DQN(num_states, num_actions).to(device)
        #policydqn.load_state_dict(torch.load(self.model, weights_only=True))

        if is_training:
            rb = ReplayBuffer(RB_SIZE)

            epsilon = self.epsilon_init
            epsilon_history = []

            print("Creating target DQN")
            targetdqn = DQN(num_states, num_actions).to(device)
            targetdqn.load_state_dict(policydqn.state_dict())

            step_count = 0

            best_reward = self.best_reward
        else:

            policydqn.eval()

        # Loops number of episodes
        print("Starting the loop")
        for episode in range(self.episodes):
            # Get state and convert to tensor
            state, _ = env.reset()
            state = self.process_state(state)

            terminated = False
            episode_reward = 0.0

            # Loop under one episode
            while not terminated:
                env.render()
                cv2.waitKey(1)

                # Epsilon Greedy Method to get the action
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device).item()

                else:
                    action = torch.argmax(policydqn(state.unsqueeze(dim=0))).squeeze().item()

                # Step with the action
                new_state, reward, terminated, truncated, info = env.step(action)

                # Calculate bumpiness and add to reward
                if self.mode != 3:
                    bumpy = self.bump(new_state['board'])
                    reward += -0.1 * bumpy

                # Convert new_state and reward to tensors
                new_state = self.process_state(new_state)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                # Add to episode of this reward
                episode_reward += reward

                # Add to the replay buffer if training and increment count
                if is_training:
                    rb.add((state, action, reward, new_state, terminated))
                    step_count += 1

                # Set state to new state
                state = new_state
            
            # After the while loop in the episode:
            # Add to rewards list
            rewards_per_episode.append(episode_reward)

            if is_training:

                # If enough replay experience has been collected
                if len(rb) > MINI_BATCH_SIZE:
                    # Get batch
                    sample = rb.sample(MINI_BATCH_SIZE)
                    self.optimize(sample, policydqn, targetdqn)

                    # Update epsilon for next episode
                    epsilon = max(epsilon * EPISILON_DECAY, EPSILON_MIN)

                    # Add epsilon to epsilon history list
                    epsilon_history.append(epsilon)

                    if step_count > NETWORK_SYNC_RATE:
                        targetdqn.load_state_dict(policydqn.state_dict())
                        step_count = 0
                    
                    #print(f"Episode: {episode}, Epsilon: {epsilon:0.3f}, Reward: {episode_reward:0.3f}")

                # If new best reward
                if episode_reward > best_reward:

                    # Save the model
                    log_message = f"New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    torch.save(policydqn.state_dict(), self.model)
                    best_reward = episode_reward

                    # Update the best reward file
                    best_reward_file = open(self.best_reward_file, "w")
                    best_reward_file.write(str(best_reward.item()))
                    best_reward_file.close()

    def optimize(self, mini_batch, policydqn, targetdqn):

        # Get values from mini batch
        states, actions, rewards, new_states, terminations = mini_batch
        terminations = terminations.to(device)

        # Calculate target q values
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * DISCOUNT_FACTOR * targetdqn(new_states).max(dim=1)[0]
        
        # Calculate current q values
        current_q = policydqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Loss function is mean squared error
        # Can be replaced by something else
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(current_q, target_q)

        optimizer = torch.optim.Adam(policydqn.parameters(), lr=LEARNING_RATE_A)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
