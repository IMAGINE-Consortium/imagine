import logging as log
from .prior import GeneralPrior, scipyPrior
from imagine.tools.icy_decorator import icy
from scipy.stats import norm
import numpy as np

@icy
class FlatPrior(GeneralPrior):
    """
    Prior distribution stating that any parameter values
    within the valid interval have the same prior probability.
    
    No initialization is required.
    """
    def __init__(self, interval=[0,1]):
        # Updates ranges
        super().__init__(interval=interval)
        # Constant pdf (for illustration)
        self._pdf = lambda x: np.ones_like(x)
    
    def __call__(self, cube):
        """
        Return variable value as it is

        parameters
        ----------
        cube : list
            List of variable values in range [0,1]

        Returns
        -------
        List of variable values in range [0,1]
        """
        log.debug('@ flat_prior::__call__')
        return cube

@icy
class GaussianPrior(scipyPrior):
    """
    Truncated normal prior distribution
    
    
    
    Parameters
    ----------
    mu : float
        The position of the mode (mean, if the truncation is symmetric) 
        of the Gaussian
    sigma : float
        Width of the distribution (standard deviation, if there was no tuncation)
    interval : tuple or list
        A pair of points representing, respectively, the minimum and maximum 
        parameter values to be considered.
    """
    def __init__(self, mu=0.0, sigma=1.0, interval=[-1.0,1.0]):
        super().__init__(distr=norm, loc=mu, scale=sigma, interval=interval)