import torch
import torch.nn as nn
import torch.optim as optim
import numpy

# DQN class responsible for forward propogation
class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        # Creates the input layer connection from the dimensions of the input to 
        # 128 neurons in the hidden layer (good practice to use this i guess)
        self.input_layer = nn.Linear(input_dim, 128)

        # One hidden layer
        self.hidden_layer1 = nn.Linear(128, 128)

        # Output layer
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        return self.output_layer(x)  # Output Q-values
    
if __name__ == "__main__":
    input_dim = 216
    output_dim = 8
    
    net = DQN(input_dim, output_dim)

    state = torch.randn(5,input_dim)
    print(state)

    q_values = net(state)
    print(q_values)

    random_action = torch.argmax(q_values).item()
    print(random_action)
    