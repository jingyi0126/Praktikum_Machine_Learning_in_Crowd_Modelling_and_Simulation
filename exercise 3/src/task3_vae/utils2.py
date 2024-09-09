import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model2 import VAE
import numpy.typing as npt


# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def reconstruction_loss(
    x_reconstructed: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> np.float64:
    """Compute the reconstruction loss.

    Args:
        x_reconstructed (npt.NDArray[np.float64]): Reconstructed data
        x (npt.NDArray[np.float64]): raw/original data

    Returns:
        np.float64: reconstruction loss
    """

    BCE = F.binary_cross_entropy_loss(x_reconstructed, x, reduction="sum")
    return BCE


def kl_loss(
    logvar: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
) -> np.float64:
    """Compute the Kullback-Leibler (KL) divergence loss using the encoded data into the mean and log-variance.

    Args:
        logvar (npt.NDArray[np.float64]): log of variance (from the output of the encoder)
        mu (npt.NDArray[np.float64]): mean (from the output of the encoder)

    Returns:
        np.float64: KL loss
    """

    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KL


# Function to compute ELBO loss
def elbo_loss(
    x: npt.NDArray[np.float64],
    x_reconstructed: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    logvar: npt.NDArray[np.float64],
    fire_evac: bool,
):
    """Compute Evidence Lower BOund (ELBO) Loss by combining the KL loss and reconstruction loss.

    Args:
        x (npt.NDArray[np.float64]): raw/original data
        x_reconstructed (npt.NDArray[np.float64]): Reconstructed data
        mu (npt.NDArray[np.float64]): mean (from the output of the encoder)
        logvar (npt.NDArray[np.float64]): log of variance (from the output of the encoder)
        fire_evac (bool): flag to indicate if output should be between [-1,1] (for coordinates)

    Returns:
        np.float64: ELBO loss (or MSE in case of coordinates for task 4)
    """

    MSE = F.mse_loss(
        x_reconstructed, x.view(x_reconstructed.shape), reduction="sum"
    )

    if fire_evac:
        return MSE
    BCE = F.binary_cross_entropy(
        x_reconstructed, x.view(x_reconstructed.shape), reduction="sum"
    )
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KL


# Function for training the VAE
def train_epoch(
    model: object, optimizer: object, dataloader: object, device
) -> np.float64:
    """Train the vae for one epoch and return the training loss on the epoch.

    Args:
        model (object): The model (of class VAE)
        optimizer (object): Adam optimizer (from torch.optim)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu') on which the training is to be done.

    Returns:
        np.float64: training loss
    """
    model.train()
    total_loss = 0
    for data, _ in dataloader:
        if (
            not model.fire_evac
        ):  # Reshape the data only for images, not for coordinates.
            data = data.view(
                -1, int(np.shape(data)[-1] * np.shape(data)[-2])
            ).to(device)
        #  Set gradient to zero! You can use optimizer.zero_grad()!
        optimizer.zero_grad()
        #  Perform forward pass of the VAE
        rec_data, mu, logvar = model.forward(data)
        #  Compute ELBO loss
        loss = elbo_loss(data, rec_data, mu, logvar, model.fire_evac)
        #  Compute gradients
        loss.backward()
        #  Perform an optimization step
        optimizer.step()
        #  Compute total_loss and return the total_loss/len(dataloader.dataset)
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader.dataset)
    return np.float64(average_loss)


def evaluate(model: object, dataloader: object, device) -> np.float64:
    """Evaluate the model on the test data and return the test loss.

    Args:
        model (object): The model (of class VAE)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').

    Returns:
        np.float64: test loss.
    """

    # return total_loss/len(dataloader.dataset)
    model.eval()
    total_loss = 0
    for data, _ in dataloader:
        if (
            not model.fire_evac
        ):  # Reshape the data only for images, not for coordinates.
            data = data.view(
                -1, int(np.shape(data)[-1] * np.shape(data)[-2])
            ).to(device)
        x_hat, mu, logvar = model(data)
        loss = elbo_loss(data, x_hat, mu, logvar, model.fire_evac)

        total_loss += loss.item()
    average_loss = total_loss / len(dataloader.dataset)
    return np.float64(average_loss)


def latent_representation(
    model: object, dataloader: object, device, epoch: int
) -> None:
    """Plot the latent representation of the data.

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
    """

    plt.figure()
    for i, (x, y) in enumerate(dataloader):
        x = x.view(-1, 784).to(device)
        mu, logvar = model.encode_data(x)
        mu = (
            mu.cpu().detach().numpy()
        )  # Since plotting with plt requires NumPy arrays, the tensor must be on the CPU.
        plt.scatter(mu[:, 0], mu[:, 1], c=y, cmap="tab10")
    plt.colorbar()
    plt.title("Latent Representation of mu")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    # plt.show()
    plt.savefig(f"pictures/Latent Representation of epoch {epoch + 1}.png")


# Function to plot reconstructed digits
def reconstruct_digits(
    model: object, dataloader: object, device, epoch: int, num_digits: int = 15
) -> None:
    """Plot reconstructed digits.

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
        num_digits (int, optional): No. of digits to be re-constructed. Defaults to 15.
    """

    for batch in dataloader:
        x, y = batch
        break
    x = x.view(-1, int(np.shape(x)[-1] * np.shape(x)[-2])).to(device)
    rec_data, mu, logvar = model(x)
    x = x.view(-1, 28, 28).cpu().numpy()
    rec_data = rec_data.view(-1, 28, 28).cpu().detach().numpy()

    # Create a figure with 2 subplots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Original Digits
    original_axes = [
        axes[0].inset_axes([j / 5, (2 - i % 3) / 3, 0.2, 0.2])
        for i in range(num_digits)
        for j in range(5)
    ]
    for i, ax in enumerate(original_axes):
        ax.imshow(x[i])  # Display the image
        ax.axis("off")
    axes[0].set_title("Original Digits")
    axes[0].axis("off")

    # Reconstructed Digits
    reconstructed_axes = [
        axes[1].inset_axes([j / 5, (2 - i % 3) / 3, 0.2, 0.2])
        for i in range(num_digits)
        for j in range(5)
    ]
    for i, ax in enumerate(reconstructed_axes):
        ax.imshow(rec_data[i])  # Display the image
        ax.axis("off")
    axes[1].set_title("Reconstructed Digits")
    axes[1].axis("off")

    # Adjust layout and add a main title
    fig.suptitle(f"Digits Comparison of epoch {epoch + 1}")

    # Save the figure
    plt.savefig(f"pictures/Digits Comparison of epoch {epoch + 1}.png")


# Function to plot generated digits
def generate_digits(model: object, epoch: int, num_samples: int = 15) -> None:
    """Generate 'num_samples' digits.

    Args:
        model (object): The model (of class VAE).
        num_samples (int, optional): No. of samples to be generated. Defaults to 15.
    """

    z = model.generate_data(num_samples)
    z = z.view(-1, 28, 28).cpu().detach().numpy()

    fig, axes = plt.subplots(3, 5, figsize=(8, 4))
    for i in range(num_samples):
        row = i // 5  # Calculate the row index
        col = i % 5  # Calculate the column index
        axes[row, col].imshow(z[i])  # Display the image
        axes[row, col].axis("off")
    plt.suptitle(f"Generated Digits when d_latent = {model.d_latent}")
    # plt.show()
    plt.savefig(
        f"pictures/Generated Digits of epoch {epoch + 1} for {model.d_latent}-dim.png"
    )


# Function to plot the loss curve
def plot_loss(train_losses, test_losses, filename):
    epochs = len(train_losses)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train")
    plt.plot(range(1, epochs + 1), test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO Loss")
    plt.title("Loss Curve")
    plt.legend()
    # plt.show()
    plt.savefig(filename)


def training_loop(
    vae: object,
    optimizer: object,
    train_loader: object,
    test_loader: object,
    epochs: int,
    plots_at_epochs: list,
    device,
) -> tuple[list, list]:
    """Train the vae model.

    Args:
        vae (object): The model (of class VAE).
        optimizer (object): Adam optimizer (from torch.optim).
        train_loader (object): A data loader that combines the training dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        test_loader (object): A data loader that combines the test dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        epochs (int): No. of epochs to train the model.
        plots_at_epochs (list): List of integers containing epoch numbers at which the plots are to be made.
        device: The device (e.g., 'cuda' or 'cpu').

    Returns:
        tuple [list, list]: Lists train_losses, test_losses containing train and test losses at each epoch.
    """
    # Lists to store the training and test losses
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        #  Compute training loss for one epoch
        train_loss = train_epoch(
            model=vae,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
        )
        #  Evaluate loss on the test dataset
        test_loss = evaluate(model=vae, dataloader=test_loader, device=device)
        #  Append train and test losses to the lists train_losses and test_losses respectively
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}"
        )

        #  For specific epoch numbers described in the worksheet, plot latent representation, reconstructed digits, generated digits after specific epochs
        if not vae.fire_evac:
            if epoch in plots_at_epochs:
                if vae.d_latent == 2:
                    latent_representation(vae, test_loader, device, epoch)
                    reconstruct_digits(vae, test_loader, device, epoch)
                generate_digits(vae, epoch)
            else:
                print("epoch is not in the list: plots_at_epochs")

    #  return train_losses, test_losses
    return train_losses, test_losses


def instantiate_vae(d_in, d_latent, d_hidden_layer, device, fire_evac=None):
    """Instantiate the variational autoencoder.

    Args:
        d_in (int): Input dimension.
        d_latent (int): Latent dimension.
        d_hidden_layer (int): Number of neurons in each hidden layer of the encoder and decoder.
        device: e.g., 'cuda' or 'cpu'. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        fire_evac: flag used to determine if the model should be use for coodinates (fire evaluation scenario) or images

    Returns:
        object: An object of class VAE
    """
    if fire_evac:
        return VAE(d_in, d_latent, d_hidden_layer, device, fire_evac).to(
            device
        )
    return VAE(d_in, d_latent, d_hidden_layer, device).to(device)


def compute_number_at_entrance(data: npt.NDArray[np.float64]) -> int:
    """Given a dataset containing coordinates of people, computes people inside the region of interest (entrance) bounded by x1, x2, y1, y2

    Args:
        data (npt.NDArray[np.float64]): array containing the coordinates of all people in the building

    Returns:
        int: number of people inside the region of interest (entrance)
    """
    x1, x2 = 130, 150
    y1, y2 = 50, 70
    condition = (
        (data[:, 0] >= x1)
        & (data[:, 0] <= x2)
        & (data[:, 1] >= y1)
        & (data[:, 1] <= y2)
    )
    at_entrance = data[condition]
    return np.sum(at_entrance)


def compute_critical_number(
    model: object,
    crit_num: int,
    scaling_coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """

    Args:
        model (object): trained VAE model to create new synthetic (generated) data of people coodinates inside building
        crit_num (int): number of critical people that can be inside the region of interest (entrance)
        scaling_coords (tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]): tuple of coordinates to transform data back and forward to [-1,1] range for training convergence

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: vectors containing number of people inside the building and in the entrance
    """
    min_coord, max_coord = scaling_coords
    at_entrance = []
    num_people = np.arange(crit_num, crit_num * 10)
    for i in num_people:
        crit_data = model.generate_data(i)
        crit_data = crit_data.detach().numpy()
        crit_data = ((crit_data + 1) * (max_coord - min_coord) / 2) + min_coord
        at_entrance.append(compute_number_at_entrance(crit_data))

    return (num_people, at_entrance)
