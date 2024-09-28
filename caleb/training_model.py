import torch
import torch.nn as nn
import torch.optim as optim
import numpy

BOARD_SIZE = 20*10
CURR_PIECE_SIZE = 2*4
NEXT_PIECE_SIZE = 2*4


import torch
import torch.nn as nn

# DQN class responsible for forward propagation
class DQN(nn.Module):  # Changed from nn.modules to nn.Module
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        # Creates the input layer connection from the dimensions of the input to 
        # 128 neurons in the hidden layer
        self.input_layer = nn.Linear(input_dim, 128)

        # One hidden layer
        self.hidden_layer1 = nn.Linear(128, 128)

        # Output layer
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        return self.output_layer(x)  # Output Q-values
