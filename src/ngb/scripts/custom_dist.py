"""
Custom LogNormal Distribution for NGBoost with Manual Log-Likelihood Implementation

This implementation provides a custom LogNormal distribution for NGBoost where
the negative log-likelihood (NLL) is manually implemented instead of relying
on scipy's built-in logpdf function.

Author: Custom implementation for ML assignment
"""

import numpy as np
from scipy.stats import lognorm as dist_scipy
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class CustomLogNormalLogScore(LogScore):
    """Custom LogScore implementation for LogNormal distribution.
    
    The log-likelihood is manually implemented here instead of using
    scipy's built-in logpdf function.
    """
    
    def score(self, Y):
        """Compute the negative log-likelihood (NLL) manually.
        
        The negative log-likelihood is the negative of this.
        
        Parameters:
        -----------
        Y : array-like
            Observed values
            
        Returns:
        --------
        nll : array
            Negative log-likelihood for each observation
        """
        log_y = np.log(Y)
        
        term1 = -log_y
        term2 = -np.log(self.scale)
        term3 = -0.5 * np.log(2 * np.pi)
        
        standardized = (log_y - self.loc) / self.scale
        term4 = -0.5 * (standardized ** 2)
        
        nll = -(term1 + term2 + term3 + term4)
        
        return nll
    
    def d_score(self, Y):
        """
        Compute the gradient of the negative log-likelihood.
        
        This is the derivative with respect to the internal parameters:
        - params[0] = loc
        - params[1] = log of scale
        
        Parameters:
        -----------
        Y : array-like
            Observed values
            
        Returns:
        --------
        D : ndarray of shape (len(Y), 2)
            Gradient of NLL with respect to [μ, log(scale)]
        """
        log_y = np.log(Y)
        D = np.zeros((len(Y), 2))
        
        D[:, 0] = (self.loc - log_y) / (self.scale ** 2)
        D[:, 1] = 1 - ((self.loc - log_y) ** 2) / (self.scale ** 2)
        
        return D
    
    def metric(self):
        """
        Compute the Fisher Information Matrix for natural gradient descent.
        
        For LogNormal distribution with parameters [μ, log(scale)]:
        Fisher[0, 0] = 1/σ²  
        Fisher[1, 1] = 2      
        Fisher[0, 1] = Fisher[1, 0] = 0
        
        Returns:
        --------
        FI : ndarray of shape (n_samples, 2, 2)
            Fisher Information Matrix for each sample
        """
        FI = np.zeros((self.scale.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / (self.scale ** 2)
        FI[:, 1, 1] = 2
        return FI


class CustomLogNormal(RegressionDistn):
    """Custom LogNormal distribution for NGBoost with manual log-likelihood.
    
    This distribution models Y|X ~ LogNormal(μ, σ²) where:
    - μ (loc): location parameter (mean of log(Y))
    - sigma (scale): scale parameter (std of log(Y))
    
    Internally, NGBoost works with parameters [μ, log(sigma)] for numerical stability.
    """

    n_params = 2
    scores = [CustomLogNormalLogScore]
    
    def __init__(self, params):
        """Initialize the distribution with internal parameters.
        
        Parameters:
        -----------
        params : array of shape (n_samples, 2)
            params[0] = location
            params[1] = log of scale
        """
        self._params = params
        self.loc = params[0]
        self.scale = np.exp(params[1])

        self.dist = dist_scipy(s=self.scale, scale=np.exp(self.loc))
        self.eps = 1e-5
    
    @staticmethod
    def fit(Y):
        """Initialize distribution parameters from data.
        
        Parameters:
        -----------
        Y : array-like
            Target values (must be positive)
            
        Returns:
        --------
        params : array of shape (2,)
            Initial parameters [μ, log(sigma)]
        """
        log_Y = np.log(Y)
        mu = np.mean(log_Y)
        sigma = np.std(log_Y)
        
        return np.array([mu, np.log(sigma)])
    
    def sample(self, m):
        """Sample from the distribution.
        
        Parameters:
        -----------
        m : int
            Number of samples
            
        Returns:
        --------
        samples : array of shape (m,)
            Random samples from the distribution
        """
        return np.array([self.dist.rvs() for i in range(m)])
    
    def __getattr__(self, name):
        """Delegate attribute access to scipy distribution for convenience methods.
        This allows us to call methods like .mean(), .var(), .cdf(), etc.
        """
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None
    
    @property
    def params(self):
        """Return user-facing parameters.
        
        Returns:
        --------
        params : dict
            Dictionary with 's' (scale) and 'scale' (exp(loc))
            following scipy's lognorm convention
        """
        return {"s": self.scale, "scale": np.exp(self.loc)}
