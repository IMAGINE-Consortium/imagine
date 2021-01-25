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
    Prior distribution where any parameter values within the valid interval
    have the same prior probability.

    Parameters
    ----------
    xmin, xmax : float
        A pair of points representing, respectively, the minimum/maximum
        parameter values to be considered.
    unit : astropy.units.Unit, optional
        If present, sets the units used for this parameter. If absent, this
        is inferred from `xmin` and `xmax`.
    wrapped : bool
        Specify whether the parameter is periodic (i.e. the range is supposed
        to "wrap-around").

    """
    def __init__(self, xmin, xmax, unit=None, wrapped=False):
        # Updates ranges
        super().__init__(xmin=xmin, xmax=xmax, wrapped=wrapped, unit=unit)
        # Computes this from `range`, after the base Prior class has
        # already dealt with units
        self.vol = self.range[1] - self.range[0]
        # Constant pdf (for illustration)
        self._pdf = lambda x: np.ones(x.shape)/self.vol.value

    def __call__(self, cube):
        log.debug('@ flat_prior::__call__')

        unit, [cube_val] = unit_checker(self.unit, [cube])
        # Rescales to the correct interval
        cube_val = cube_val * (self.range[1].value -  self.range[0].value)
        cube_val += self.range[0].value

        return cube_val << unit


class GaussianPrior(ScipyPrior):
    """
    Normal prior distribution.

    This can operate either as a regular Gaussian distribution
    (defined from -infinity to infinity) or, if `xmin` and `xmax` values
    are set, as a trucated Gaussian distribution.

    Parameters
    ----------
    mu : float
        The position of the mode (mean, if the truncation is symmetric)
        of the Gaussian
    sigma : float
        Width of the distribution (standard deviation, if there was no tuncation)
    xmin, xmax : float
        A pair of points representing, respectively, the minimum/maximum
        parameter values to be considered (i.e. the truncation interval).
        If these are not provided (or set to `None`), the prior range is
        assumed to run from -infinity to infinity
    unit : astropy.units.Unit, optional
        If present, sets the units used for this parameter. If absent, this
        is inferred from `mu` and `sigma`.
    wrapped : bool
        Specify whether the parameter is periodic (i.e. the range is supposed
        to "wrap-around").

    """

    def __init__(self, mu=None, sigma=None, xmin=None, xmax=None, unit=None,
                 wrapped=False, **kwargs):

        assert mu is not None, 'A value for mu must be provided'
        assert sigma is not None, 'A value for sigma must be provided'

        unit, [mu_val, sigma_val] = unit_checker(unit, [mu, sigma])

        super().__init__(distr=norm, loc=mu, scale=sigma, unit=unit,
                         xmin=xmin, xmax=xmax, **kwargs)
