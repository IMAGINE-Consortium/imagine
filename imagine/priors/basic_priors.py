# %% IMPORTS
# Built-in imports
import logging as log

# Package imports
import numpy as np
from scipy.stats import norm
import astropy.units as u

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
    def __init__(self, xmin, xmax, unit=None):
        # Updates ranges
        super().__init__(xmin=xmin, xmax=xmax, unit=unit)
        self.vol = xmax - xmin
        # Constant pdf (for illustration)
        self._pdf = lambda x: np.ones_like(x)/self.vol.value

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
        return cube*self.vol.value + self.range[0].value


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
    def __init__(self, unit, mu=0.0, sigma=1.0):
        assert isinstance(unit, u.Unit)
        super().__init__(distr=norm, loc=mu, scale=sigma, unit=unit)
