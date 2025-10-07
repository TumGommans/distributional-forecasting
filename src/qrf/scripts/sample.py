"""Script for sampling observations from quantiles."""

import numpy as np
from scipy.interpolate import interp1d

def sample(
    quantiles: np.ndarray,
    quantile_preds: np.ndarray, 
    n_samples: int=10000
) -> np.ndarray:
    """Generate random samples from predicted quantiles.

    Args:
        quantile_preds: array of predicted quantiles for each observation
        n_samples: the number of samples to generate per observation

    Returns:
        np.ndarray: array of generated samples, shape (n_observations, n_samples)
    """
    n_observations = quantile_preds.shape[0]
    samples = np.zeros((n_observations, n_samples))
    
    uniform_samples = np.random.uniform(size=(n_observations, n_samples))

    for i in range(n_observations):
        interpolator = interp1d(
            quantiles,
            quantile_preds[i],
            bounds_error=False,
            fill_value=(quantile_preds[i, 0], quantile_preds[i, -1])
        )
        samples[i, :] = interpolator(uniform_samples[i, :])
        
    return samples