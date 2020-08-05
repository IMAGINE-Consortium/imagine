# %% IMPORTS
# Built-in imports
import logging as log

# Package imports
import numpy as np
from scipy.stats import norm

# IMAGINE imports
from imagine.priors import GeneralPrior, ScipyPrior

# All declaration
__all__ = ['FlatPrior', 'GaussianPrior']


# %% CLASS DEFINITIONS
class FlatPrior(GeneralPrior):
    """
    Prior distribution stating that any parameter values
    within the valid interval have the same prior probability.

    No initialization is required.
    """
    def __init__(self, interval):
        # Updates ranges
        super().__init__(interval=interval)
        self.vol = interval[1] - interval[0]
        # Constant pdf (for illustration)
        self._pdf = lambda x: np.ones_like(x)/self.vol

    def __call__(self, cube):
        """
        Return uniformly distributed variable

        parameters
        ----------
        cube : list
            List of variable values in range [0,1]

        Returns
        -------
        List of variable values in the given interval
        """
        log.debug('@ flat_prior::__call__')
        return cube*self.vol + self.range[0]


class GaussianPrior(ScipyPrior):
    """
    Normal prior distribution



    Parameters
    ----------
    mu : float
        The position of the mode (mean, if the truncation is symmetric)
        of the Gaussian
    sigma : float
        Width of the distribution (standard deviation, if there was no tuncation)
    """
    def __init__(self, mu=0.0, sigma=1.0):
        super().__init__(distr=norm, loc=mu, scale=sigma)