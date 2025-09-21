# src/loss/mdn_loss.py

import torch
import math

def log_normal_mdn_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the negative log-likelihood loss for a Log-Normal Mixture Density Network.

    This function expects the raw, positive target `y` and calculates the likelihood
    using the Log-Normal probability density function.

    Args:
        pi (torch.Tensor): Mixture weights, shape (batch_size, n_gaussians).
        mu (torch.Tensor): The means of the underlying Normal distributions for log(y).
        sigma (torch.Tensor): The stds of the underlying Normal distributions for log(y).
        y (torch.Tensor): The original, positive target values, shape (batch_size, 1).

    Returns:
        torch.Tensor: The mean negative log-likelihood loss.
    """
    # The Log-Normal PDF is: 1/(y*sigma*sqrt(2pi)) * exp(-(log(y)-mu)^2 / (2*sigma^2))
    # Add a small epsilon for numerical stability with log(y)
    log_y = torch.log(y + 1e-8)

    # Expand y and log_y to be broadcastable with mu and sigma
    y_exp = y.expand_as(mu)
    log_y_exp = log_y.expand_as(mu)

    # Calculate the exponent term
    exponent = -0.5 * ((log_y_exp - mu) / sigma) ** 2
    
    # Calculate the coefficient, including the 1/y term
    coeff = 1.0 / (y_exp * sigma * math.sqrt(2 * math.pi))

    # Calculate the PDF for each component
    component_pdfs = coeff * torch.exp(exponent)
    
    # Calculate the likelihood of the mixture
    likelihood = torch.sum(pi * component_pdfs, dim=1)
    
    # Calculate the negative log-likelihood loss
    nll = -torch.log(likelihood + 1e-8)
    
    return torch.mean(nll)


def gaussian_mdn_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the negative log-likelihood loss for a Gaussian Mixture Density Network.

    Args:
        pi (torch.Tensor): Mixture weights, shape (batch_size, n_gaussians).
        mu (torch.Tensor): Means of the Gaussian components, shape (batch_size, n_gaussians).
        sigma (torch.Tensor): Standard deviations of the Gaussian components,
                              shape (batch_size, n_gaussians).
        y (torch.Tensor): The target values, shape (batch_size, 1).

    Returns:
        torch.Tensor: The mean negative log-likelihood loss, a scalar tensor.
    """
    # Calculate the Gaussian probability density function
    # The target y has shape (batch_size, 1), we need to make it
    # broadcastable with mu and sigma of shape (batch_size, n_gaussians).
    y = y.expand_as(mu)

    # N(y|mu, sigma^2) = (1/sqrt(2*pi*sigma^2)) * exp(-(y-mu)^2 / (2*sigma^2))
    exponent = -0.5 * ((y - mu) / sigma) ** 2
    normal_pdf = (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(exponent)

    # Calculate the likelihood of the mixture
    # P(y|x) = sum_{k=1 to K} pi_k * N(y|mu_k, sigma_k^2)
    likelihood = torch.sum(pi * normal_pdf, dim=1)

    # Add a small epsilon for numerical stability to avoid log(0)
    epsilon = 1e-8
    
    # Calculate the negative log-likelihood loss
    nll = -torch.log(likelihood + epsilon)

    return torch.mean(nll)