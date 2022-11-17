# %% IMPORTS
# Built-in imports
import abc
import collections
import logging as log

# Package imports
import astropy.units as u
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d

# IMAGINE imports
from imagine.tools import BaseClass, req_attr
from imagine.tools import unit_checker

from imagine.fields.TOGOModel import Model
from imagine.grid.TOGOGrid import ParameterSpace, ParameterSpaceDict, ScalarSpace

# All declaration
__all__ = ['Prior', 'CustomPrior', 'MultivariatePrior']


class Prior(Model):
    def __init__(self, pd_info, parameter_name, unit=None, pdf_n_points=1000, numerical_range=None, **distr_kwargs):
        if unit is None:
            unit = u.Unit(s='')
        param_space = ParameterSpace(1, parameter_name)
        super().__init__(param_space, param_space, call_by_method=True)
        self._pdf_n_points = pdf_n_points
        self._numerical_range = numerical_range

        self.set_distr(pd_info, distr_kwargs)

        self._unit = unit

    @property
    def unit(self):
        return self._unit

    def draw_sample(self):
        return self._distr.rvs() << self.unit

    def compute_model(self, parameters):
        return self._distr.ppf(u) << self.unit

    def set_distr(self, f, distr_kwargs):
        if isinstance(f, stats.rv_continuous):
            self._distr = getattr(stats, f)(**distr_kwargs)
        elif isinstance(f, str):
            self._distr = getattr(stats, f)(**distr_kwargs)

        elif isinstance(f, (collections.Callable, np.ndarray,)):
            if isinstance(f, np.nd_array):
                raise NotImplementedError
            x = np.linspace(*self._numerical_range, self._pdf_n_points)
            cdf_y = np.append(0, np.cumsum(x))
            cdf_y /= cdf_y.max()  # Slight rescaling to avoid potential problems due to truncation
            cdf = interp1d(x, cdf_y)
            ppf = interp1d(cdf_y, x)

            class Distr(stats.rv_continuous):

                def _pdf(self, x):
                    return f(x)

                def _cdf(self, x):
                    return cdf(x)

                def _ppf(self, x):
                    return ppf(x)

            self._distr = Distr(**distr_kwargs)
        else:
            raise TypeError('Prior must me set with either a scipy distr object, string indicating a scipy distr object or a callable for the pdf')

# %% CLASS DEFINITIONS


class MultivariatePrior(Model):

    def __init__(self, dict_of_priors, correlator=None):

        self.correlator = correlator
        self.dict_of_priors = dict_of_priors
        param_space = ParameterSpaceDict({n: ScalarSpace(n) for n in dict_of_priors})
        super().__init__(param_space, param_space, call_by_method=True)

    def draw_sample(self):
        return {n: prior.draw_sample() for n, prior in self.dict_of_priors.items()}

    def compute_model(self, parameters):
        return {n: prior(parameters[n]) for n, prior in self.dict_of_priors.items()}


class CustomPrior(Prior):
    """
    Allows constructing a prior from a pre-existing sampling of the parameter
    space or a known probability density function (PDF).

    Parameters
    ----------
    samples : array_like
        Array containing a sample of the prior distribution one wants to use.
        Note: this will use :py:class:`scipy.stats.gaussian_kde` to compute
        the probability density function (PDF) through kernel density
        estimate using Gaussian kernels.
    pdf_fun : function
        A Python function containing the PDF for this prior.
        Note that the  function must be able to operate on
        :py:obj:`Quantity <astropy.units.quantity.Quantity>` object if the
        parameter is not dimensionless.
    xmin, xmax : float
        A pair of points representing, respectively, the minimum/maximum
        parameter values to be considered. If not provided (or set to `None`),
        the smallest/largest value in the sample minus/plus one standard
        deviation will be used.
    unit : astropy.units.Unit
        If present, sets the units used for this parameter. If absent, this
        is inferred from `xmin` and `xmax`.
    wrapped : bool
        Specify whether the parameter is periodic (i.e. the range is supposed
        to "wrap-around").
    bw_method: scalar or str
        Used by :py:class:`scipy.stats.gaussian_kde` to select the bandwidth
        employed to estimate the PDF from provided samples. Can be a number,
        if using fixed bandwidth, or strings ‘scott’ or ‘silverman’ if these
        rules are to be selected.
    pdf_npoints : int
        Number of points used to evaluate pdf_fun or the KDE constructed from
        the samples.
    samples_ref : bool
        If True (default), a reference to the samples is stored, allowing prior
        correlations to be computed by the Pipeline.
    """

    def __init__(self, samples=None, pdf_fun=None, xmin=None, xmax=None,
                 unit=None, wrapped=False, bw_method=None, pdf_npoints=1500,
                 samples_ref=True):
        # If needed, constructs a pdf function from samples, using KDE
        if samples is not None:
            unit, [xmin_val, xmax_val, samples_val] = unit_checker(unit, [xmin, xmax, samples])
            assert unit is not None, 'At least one input must have a unit or a astropy unit must be provided'
            if (xmin is None) or (xmax is None):
                std = np.std(samples_val)
            if xmin is None:
                xmin_val = samples_val.min() - std
            if xmax is None:
                xmax_val = samples_val.max() + std

            pdf_fun = stats.gaussian_kde(samples_val, bw_method=bw_method)
        else:
            assert (xmin is not None and xmax is not None), 'both xmin and xmax must be given for pdf_fun'
            unit, [xmin_val, xmax_val] = unit_checker(unit, [xmin, xmax])

        # Evaluates the PDF
        pdf_x = np.linspace(xmin_val, xmax_val, pdf_npoints)

        if unit is not None:
            pdf_y = pdf_fun(pdf_x << unit)
        else:
            pdf_y = pdf_fun(pdf_x)

        super().__init__(xmin=xmin_val, xmax=xmax_val, unit=unit,
                         wrapped=wrapped, pdf_npoints=pdf_npoints)
        dx = (xmax_val - xmin_val) / pdf_npoints
        inv_norm = pdf_y.sum() * dx
        pdf_y = (pdf_y / inv_norm)

        self._pdf_y = pdf_y
        self._pdf_x = pdf_x

        if (samples is not None) and samples_ref:
            # If requested, stores the samples to allow later (in the Pipeline)
            # determination of correlations between parameters
            self.samples = samples << self.unit  # Adjusts units to avoid errors
