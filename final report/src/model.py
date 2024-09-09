import torch
import torch.nn as nn
import torch.optim as optim


class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        '''
        Artificial Neural Network (ANN) model with multiple hidden layers.

        This model consists of multiple fully connected (dense) layers with ReLU activation,
        followed by a final output layer.

        Parameters:
        - input_size (int): The size of the input features.
        - hidden_sizes (list): A list of integers, where each integer represents the size of a hidden layer.
                            The length of the list determines the number of hidden layers.
        - output_size (int): The size of the output layer.
        '''
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
        '''
        Defines the forward pass of the neural network model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, output_size).

        '''
        return self.network(x)
