# src/metrics/crps.py

import torch
import scoringrules as sr

def calculate_crps(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 1000
) -> torch.Tensor:
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for a GMM
    by sampling from the distribution to create an ensemble forecast.

    Args:
        pi (torch.Tensor): Mixture weights, shape (batch_size, n_gaussians).
        mu (torch.Tensor): Means of the Gaussian components, shape (batch_size, n_gaussians).
        sigma (torch.Tensor): Standard deviations of the Gaussian components,
                              shape (batch_size, n_gaussians).
        y (torch.Tensor): The target values, shape (batch_size, 1).
        n_samples (int): The number of samples to draw for the ensemble.

    Returns:
        torch.Tensor: The mean CRPS for the batch, a scalar tensor.
    """
    
    # Generate samples from the GMM
    # 1. Choose a component for each sample based on weights (pi)
    component_indices = torch.multinomial(pi, n_samples, replacement=True) # (batch, n_samples)
    
    # 2. Gather the mu and sigma for the chosen components
    mu_samples = torch.gather(mu, 1, component_indices)
    sigma_samples = torch.gather(sigma, 1, component_indices)
    
    # 3. Generate random samples from the normal distributions
    #    torch.randn_like creates a tensor with the same shape filled with N(0,1) samples
    log_y_forecasts = mu_samples + sigma_samples * torch.randn_like(mu_samples)

    # === CRITICAL STEP ===
    # Transform the log-scale samples back to the original scale of Y
    ensemble_forecasts = torch.exp(log_y_forecasts)

    # Use scoringrules to compute CRPS
    # Ensure y is in the correct format (numpy array)
    y_numpy = y.squeeze().cpu().numpy()
    forecasts_numpy = ensemble_forecasts.cpu().numpy()
    
    # sr.crps_ensemble returns CRPS for each observation
    crps_values = sr.crps_ensemble(y_numpy, forecasts_numpy)
    
    return torch.tensor(crps_values.mean(), device=pi.device)