import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt


class VAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_latent: int,
        d_hidden_layer: int,
        device,
        fire_evac: bool = False,
    ):
        """Initialiize the VAE.

        Args:
            d_in (int): Input dimension
            d_latent (int): Latent dimension.
            d_hidden_layer (int): Number of neurons in the hidden layers of encoder and decoder.
            device: 'cpu' or 'cuda'
            fire_evac: flag used to determine if the model should be use for coodinates (fire evaluation scenario) or images
        """
        super(VAE, self).__init__()

        # Set device
        self.device = device

        #  Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
        self.d_in = d_in
        self.d_latent = d_latent
        self.d_hidden_layer = d_hidden_layer
        self.fire_evac = fire_evac
        #  Initialize the encoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).
        self.encoder = nn.Sequential(
            nn.Linear(self.d_in, self.d_hidden_layer),
            nn.LeakyReLU(),
            nn.Linear(self.d_hidden_layer, self.d_hidden_layer),
            nn.LeakyReLU(),
        )
        #  Initialize a linear layer for computing the mean (one of the outputs of the encoder)
        self.fc_mu = nn.Linear(self.d_hidden_layer, self.d_latent)
        #  Initialize a linear layer for computing the variance (one of the outputs of the encoder)
        self.fc_logvar = nn.Linear(self.d_hidden_layer, self.d_latent)
        #  Initialize the decoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).
        layers = [
            nn.Linear(self.d_latent, self.d_hidden_layer),
            nn.LeakyReLU(),
            nn.Linear(self.d_hidden_layer, self.d_hidden_layer),
            nn.LeakyReLU(),
            nn.Linear(self.d_hidden_layer, self.d_in),
        ]

        # Define the activation depending on whether images (normalized to [0,1]) or coordinates (normalized to [-1,1]) are being used.
        if self.fire_evac:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def encode_data(
        self, x: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Forward pass throguh the encoder.

        Args:
            x (npt.NDArray[np.float64]): Input data

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: mean, log of variance
        """
        #  Implement method!!
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(
        self, mu: npt.NDArray[np.float64], logvar: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Use the reparameterization trick for sampling from a Gaussian distribution.

        Args:
            mu (npt.NDArray[np.float64]): Mean of the Gaussian distribution.
            logvar (npt.NDArray[np.float64]): Log variance of the Gaussian distribution.

        Returns:
            npt.NDArray[np.float64]: Sampled latent vector.
        """
        #  Implement method!!
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode_data(
        self, z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Decode latent vectors to reconstruct data.

        Args:
            z (npt.NDArray[np.float64]): Latent vector.

        Returns:
            npt.NDArray[np.float64]: Reconstructed data.
        """
        #  Implement method!!
        rec_data = self.decoder(z)
        return rec_data

    def generate_data(self, num_samples: int) -> npt.NDArray[np.float64]:
        """Generate data by sampling and decoding 'num_samples' vectors in the latent space.

        Args:
            num_samples (int): Number of generated data samples.

        Returns:
            npt.NDArray[np.float64]: generated samples.
        """
        #  Implement method!!
        # Hint (You may need to use .to(self.device) for sampling the latent vector!)
        z = torch.rand(num_samples, self.d_latent).to(self.device)
        samples = self.decoder(z)
        return samples

    def forward(
        self, x: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Forward pass of the VAE.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: reconstructed data, mean
            of gaussian distribution (encoder), variance of gaussian distribution (encoder)
        """
        #  Implement method!!
        # print("Input shape:", x.shape)
        mu, logvar = self.encode_data(x)
        # print("Mu shape:", mu.shape, "Logvar shape:", logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print("Latent z shape:", z.shape)
        rec_data = self.decode_data(z)  # torch.transpose(z, 0, 1)
        # print("Reconstructed data shape:", rec_data.shape)
        return rec_data, mu, logvar
