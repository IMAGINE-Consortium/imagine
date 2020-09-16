# %% IMPORTS
# Built-in imports
import logging as log

# Package imports
import numpy as np
from scipy.stats import norm
import astropy.units as u

# IMAGINE imports
from imagine.priors import Prior, ScipyPrior
from imagine.tools import unit_checker
# All declaration
__all__ = ['FlatPrior', 'GaussianPrior']


# %% CLASS DEFINITIONS
class FlatPrior(Prior):
    """
    Prior distribution stating that any parameter values
    within the valid interval have the same prior probability.

    No initialization is required.
    """
    def __init__(self, xmin, xmax, unit=None):
        # Updates ranges
        super().__init__(xmin=xmin, xmax=xmax, unit=unit)
        # Computes this from `range`, after the base Prior class has
        # already dealt with units
        self.vol = self.range[1] - self.range[0]
        # Constant pdf (for illustration)
        self._pdf = lambda x: np.ones_like(x)/self.vol.value

    def __call__(self, cube):
        """
        Return uniformly distributed variable

        parameters
        ----------
        cube : list
            List of variable values

        Returns
        -------
        List of variable values in the given interval
        """
        log.debug('@ flat_prior::__call__')

        unit, [cube_val] = unit_checker(self.unit, [cube])
        # Rescales to the correct interval
        cube_val = cube_val * (self.range[1].value -  self.range[0].value)
        cube_val += self.range[0].value

        return cube_val << unit


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

    def __init__(self, mu=0.0, sigma=1.0, xmin=None, xmax=None, unit=None,
                 **kwargs):

        unit, [mu_val, sigma_val] = unit_checker(unit, [mu, sigma])

        super().__init__(distr=norm, loc=mu, scale=sigma, unit=unit,
                         xmin=xmin, xmax=xmax, **kwargs)
