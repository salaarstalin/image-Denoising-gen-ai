"""A convolutional variational autoencoder for 64x64 RGB images."""
from __future__ import annotations

import torch
from torch import Tensor, nn

IMAGE_SIZE = 64
MAX_LATENT_DIM = 128
LATENT_OPTIONS = (32, 64, 96, 128)


class ConvVAE(nn.Module):
    """Learned encoder -> Gaussian latent distribution -> learned decoder.

    At inference, posterior mean is used for deterministic reconstruction.
    A low-dimensional prefix is retained to simulate increasingly strong
    latent bottlenecks; the decoder never sees the discarded coefficients.
    """

    def __init__(self, latent_dim: int = MAX_LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 96, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(96 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(96 * 8 * 8, latent_dim)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 96 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (96, 8, 8)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden).clamp(-10, 6)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.from_latent(z))

    def forward(
        self, x: Tensor, active_dims: int = MAX_LATENT_DIM,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if active_dims < 1 or active_dims > self.latent_dim:
            raise ValueError('Invalid latent dimension')
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        # Inactive dimensions are truly removed from the decoder's input.
        if active_dims < self.latent_dim:
            z = torch.cat((z[:, :active_dims], torch.zeros_like(z[:, active_dims:])), 1)
        return self.decode(z), mu, logvar
