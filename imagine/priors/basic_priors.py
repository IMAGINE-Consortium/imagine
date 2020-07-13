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


class GaussianPrior(ScipyPrior):
    """
    (Truncated) normal prior distribution



    Parameters
    ----------
    mu : float
        The position of the mode (mean, if the truncation is symmetric)
        of the Gaussian
    sigma : float
        Width of the distribution (standard deviation, if there was no tuncation)
    interval : None, tuple or list
        If not none, a pair of points representing, respectively, the minimum and maximum
        parameter values to be considered. If none, the Gaussian is not truncated.
    """
    def __init__(self, mu=0.0, sigma=1.0, interval=None):
        super().__init__(distr=norm, loc=mu, scale=sigma, interval=interval)