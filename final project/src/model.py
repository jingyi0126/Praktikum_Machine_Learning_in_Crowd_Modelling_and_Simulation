import torch
import torch.nn as nn
import torch.optim as optim

class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ANNModel, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Add the first hidden layer, taking input_size as input
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Add subsequent hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        
        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential container
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)