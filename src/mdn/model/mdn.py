# src/model/mdn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LogNormalMDN(nn.Module):
    """
    A Gaussian Mixture Density Network (MDN) with a shared core network
    and three separate heads for predicting the parameters of the mixture
    distribution: mixture weights (pi), means (mu), and standard deviations (sigma).
    """

    def __init__(
        self,
        input_dim: int,
        n_gaussians: int,
        core_hidden_dims: Tuple[int, ...],
        head_hidden_dims: Tuple[int, ...]
    ):
        """
        Initializes the GaussianMDN model.

        Args:
            input_dim (int): The dimensionality of the input features.
            n_gaussians (int): The number of Gaussian components in the mixture.
            core_hidden_dims (Tuple[int, ...]): A tuple specifying the number of
                                                neurons in each hidden layer of the core network.
            head_hidden_dims (Tuple[int, ...]): A tuple specifying the number of
                                                neurons in each hidden layer of the individual heads.
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_gaussians = n_gaussians

        # Core network
        core_layers = []
        in_features = input_dim
        for h_dim in core_hidden_dims:
            core_layers.append(nn.Linear(in_features, h_dim))
            core_layers.append(nn.ReLU())
            in_features = h_dim
        self.core_network = nn.Sequential(*core_layers)

        # Head for mixture weights (pi)
        pi_layers = []
        in_features_head = core_hidden_dims[-1]
        for h_dim in head_hidden_dims:
            pi_layers.append(nn.Linear(in_features_head, h_dim))
            pi_layers.append(nn.ReLU())
            in_features_head = h_dim
        pi_layers.append(nn.Linear(in_features_head, n_gaussians))
        self.pi_head = nn.Sequential(*pi_layers)

        # Head for means (mu)
        mu_layers = []
        in_features_head = core_hidden_dims[-1]
        for h_dim in head_hidden_dims:
            mu_layers.append(nn.Linear(in_features_head, h_dim))
            mu_layers.append(nn.ReLU())
            in_features_head = h_dim
        mu_layers.append(nn.Linear(in_features_head, n_gaussians))
        self.mu_head = nn.Sequential(*mu_layers)

        # Head for standard deviations (sigma)
        sigma_layers = []
        in_features_head = core_hidden_dims[-1]
        for h_dim in head_hidden_dims:
            sigma_layers.append(nn.Linear(in_features_head, h_dim))
            sigma_layers.append(nn.ReLU())
            in_features_head = h_dim
        sigma_layers.append(nn.Linear(in_features_head, n_gaussians))
        self.sigma_head = nn.Sequential(*sigma_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the mixture weights (pi), means (mu), and standard deviations (sigma).
            - pi: (batch_size, n_gaussians)
            - mu: (batch_size, n_gaussians)
            - sigma: (batch_size, n_gaussians)
        """
        # Pass input through the shared core network
        core_output = self.core_network(x)

        # Calculate parameters from separate heads
        pi_logits = self.pi_head(core_output)
        pi = F.softmax(pi_logits, dim=1)

        mu = self.mu_head(core_output)

        # Use softplus for sigma to ensure positivity
        sigma = F.softplus(self.sigma_head(core_output)) + 1e-6 # Add epsilon for stability

        return pi, mu, sigma