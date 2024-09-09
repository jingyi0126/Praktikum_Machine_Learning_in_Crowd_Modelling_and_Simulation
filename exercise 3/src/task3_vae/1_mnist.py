import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import VAE
from utils import *
import torchvision

""" This script is used to train and test the VAE.
"""

############################################################
## Subtasks 3.3 & 3.4 in the worksheet ##
############################################################
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset: Train and test
train_dataset = MNIST(
    "./data", train=True, download=True, transform=ToTensor()
)
test_dataset = MNIST(
    "./data", train=False, download=True, transform=ToTensor()
)

# Set the learning rate, batch size and no. of epochs
lr = 0.001
batch_size = 128
epochs = 50
# Create an instance of Dataloader for train_dataset using torch.utils.data, use appropriate batch size, keep shuffle=True.
train_dataset = DataLoader(
    MNIST(
        "./data",
        train=True,
        transform=torchvision.transforms.Compose(
            [
                ToTensor(),
                # torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        ),
        download=True,
    ),
    batch_size=batch_size,
    shuffle=True,
)
# Create an instance of Dataloader for test_dataset using torch.utils.data, use appropriate batch size, keep shuffle=False.
test_dataset = DataLoader(
    MNIST(
        "./data",
        train=False,
        transform=torchvision.transforms.Compose(
            [
                ToTensor(),
                # torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        ),
        download=True,
    ),
    batch_size=batch_size,
    shuffle=False,
)
# Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
d_in = 784  # 28 * 28
d_latent = 2
d_hidden_layer = 256

# Instantiate the VAE model with a latent dimension of 2, using the utility function instantiate_vae() from utils
vae = instantiate_vae(d_in, d_latent, d_hidden_layer, device=device)

# Set up an appropriate optimizer from torch.optim with an appropriate learning rate
optimizer = optim.Adam(vae.parameters(), lr=lr)

plots_at_epochs = [0, 4, 24, 49]  # generate plots at epoch numbers

# Compute train and test losses by performing the training loop using the utility function training_loop() from utils

train_losses, test_losses = training_loop(
    vae,
    optimizer,
    train_dataset,
    test_dataset,
    epochs,
    plots_at_epochs,
    device=device,
)

print(train_losses, test_losses)
# Plot the loss curve using the utility function plot_loss() from utils
plot_loss(train_losses, test_losses, filename="pictures/loss_curve_2.png")

##############################################################
##### Subtask 3.5 in the worksheet #######
##############################################################
# Create the VAE model with a latent dimension of 32
# Repeat the above steps with the latent dimension of 32 and compute train and test losses
d_latent = 32
vae = instantiate_vae(d_in, d_latent, d_hidden_layer, device=device)
optimizer = optim.Adam(vae.parameters(), lr=lr)
plots_at_epochs = [0, 4, 24, 49]  # generate plots at epoch numbers

# (5a) Compare 15 generated digits using the utility function reconstruct_digits()
train_losses, test_losses = training_loop(
    vae,
    optimizer,
    train_dataset,
    test_dataset,
    epochs,
    plots_at_epochs,
    device=device,
)

# (5b) Plot the loss curve using the utility function plot_loss()
print(train_losses, test_losses)
plot_loss(train_losses, test_losses, filename="pictures/loss_curve_32.png")
