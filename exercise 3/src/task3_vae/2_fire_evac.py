import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model2 import VAE
from utils2 import *
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Set devide to gpu if device, else cpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" This script is used to train and test the VAE on the fire_evac dataset. 
(Bonus: You can simulate trajectories with Vadere, for bonus points.) 
Not included in automated tests, as it's open-ended.
"""

# Download the FireEvac dataset
# Importing train and test datasets.
train_dataset = np.load(
    os.path.join(Path(os.getcwd()).parents[1], "data/FireEvac_train_set.npy")
)
test_dataset = np.load(
    os.path.join(Path(os.getcwd()).parents[1], "data/FireEvac_test_set.npy")
)

# Convert data to float32 format to be compatible with default torch dtype used for model parameters.
train_dataset = train_dataset.astype(np.float32)
test_dataset = test_dataset.astype(np.float32)
original_train_dataset = np.copy(train_dataset)

# Make a scatter plot to visualise it.
plt.scatter(train_dataset[:, 0], train_dataset[:, 1], label="Train set")
plt.scatter(test_dataset[:, 0], test_dataset[:, 1], label="Test set")
plt.legend()
plt.title("Scatter plot of density p(x) for train and test datasets")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.savefig("data_plot.png")

# Rescale training and test datasets to [-1,1] range for better training convergence.
max_coord = train_dataset.max(axis=0)
min_coord = train_dataset.min(axis=0)

train_dataset = (train_dataset - min_coord) * 2 / (max_coord - min_coord) - 1
test_dataset = (test_dataset - min_coord) * 2 / (max_coord - min_coord) - 1

# Train a VAE on the FireEvac data
# Training parameters
lr = 0.005
batch_size = 64
epochs = 200

# Model parameters
d_in = 2
d_latent = 2
d_hidden_layer = 128

# DataLoader object creation and training
train_dataloader = DataLoader(
    list(zip(train_dataset, torch.zeros(train_dataset.shape[0], 1))),
    batch_size=batch_size,
    shuffle=True,
)
test_dataloader = DataLoader(
    list(zip(test_dataset, torch.zeros(test_dataset.shape[0], 1))),
    batch_size=batch_size,
    shuffle=False,
)

model = instantiate_vae(
    d_in, d_latent, d_hidden_layer, device=device, fire_evac=True
)
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses, test_losses = training_loop(
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    epochs,
    plots_at_epochs=[],
    device=device,
)
plot_loss(train_losses, test_losses, filename="loss_curve_fire.png")

# Make a scatter plot of the reconstructed test set
rec_data, _, _ = model.forward(torch.tensor(test_dataset))
rec_data = rec_data.detach().numpy()
rec_data = ((rec_data + 1) * (max_coord - min_coord) / 2) + min_coord

plt.figure()
plt.scatter(rec_data[:, 0], rec_data[:, 1], label="Rec. data")
plt.legend()
plt.title("Scatter plot of density p(x) for reconstructed test data")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.savefig("rec_plot.png")

# Make a scatter plot of 1000 generated samples.
gen_data = model.generate_data(1000)
gen_data = gen_data.detach().numpy()
gen_data = ((gen_data + 1) * (max_coord - min_coord) / 2) + min_coord

plt.figure()
plt.scatter(gen_data[:, 0], gen_data[:, 1], label="Gen. data")
plt.scatter(original_train_dataset[:, 0], original_train_dataset[:, 1], label="Train set")
plt.legend()
plt.title("Scatter plot of density p(x) for generated data")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.savefig("gen_plot.png")

# Generate data to estimate the critical number of people for the MI building
crit_num = 100  # Critical number of people at the building entrance
num_people, at_entrance = compute_critical_number(
    model, crit_num, (min_coord, max_coord)
)  # Vectors with number of people in building and at entrance

plt.figure()
plt.plot(num_people, at_entrance, label="People at entrance")
plt.xlabel("Num. people in building (Nb)")
plt.ylabel("Num. people at entrance (Ne)")
plt.title("Plot of Ne as a function of Nb")
plt.axhline(
    y=crit_num, color="r", linestyle="dashed", label="Crit. num. people"
)
plt.savefig("crit_plot.png")

# Write pedestrian coordinates to add pedestrians to vadere simulation.
#np.savetxt("pedestrian_pos.txt", gen_data) # Save generated data into txt file to load into vadere to modify scenario for adding pedestrians.
np.savetxt("pedestrian_pos.txt", original_train_dataset) # Saving train data to be able to generate simulation, as generated data is not correctly being generated.