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
from imagine.grid.TOGOGrid import ParameterSpace

# All declaration
__all__ = ['Prior', 'CustomPrior']


class Prior(Model):
    def __init__(self, pdf, parameter_name, unit=None, pdf_n_points=1000, numerical_range=None, **distr_kwargs):
        if unit is None:
            unit = u.Unit(s='')
        param_space = ParameterSpace(1, parameter_name)
        super().__init__(param_space, param_space, call_by_method=True)
        self._pdf_n_points = pdf_n_points
        self._numerical_range = numerical_range

        self.distr = self.set_distr(pdf, distr_kwargs)

        self._cdf = None
        self._inv_cdf = None
        self._pdf = None
        self._unit = unit

    def draw_sample(self):
        self._distr.rvs(self.n) << self.unit

    def compute_field(self, u):
        return self.inv_cdf(u) << self.unit

    @property
    def distr(self):
        """
        Constructs a scipy distribution based on an IMAGINE prior
        """

        return self._distr

    def set_distr(self, f, distr_kwargs):
        if isinstance(f, stats.rv_continuous):
            self._distr = getattr(stats, f)(**distr_kwargs)
        elif isinstance(f, str):
            self._distr = getattr(stats, f)(**distr_kwargs)

        elif isinstance(f, collections.Callable):
            xmin_val = self._numerical_range[0].value
            dx = (self._numerical_range[1].value-self._numerical_range[0].value)/self._pdf_npoints
            cdf_y = np.append(0, np.cumsum(self._pdf_y * dx))
            cdf_y /= cdf_y.max()  # Avoids potential problems due to truncation
            cdf_x = np.append(xmin_val, self._pdf_x + dx)
            cdf = interp1d(cdf_x, cdf_y)
            ppf = interp1d(cdf_y, cdf_x)

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


class MultidimensionalPrior():

    def __init__(self, dict_of_priors, correlator):

        self._parameter_names = list(dict_of_priors.keys())
        self.correlator = None
        self.dict_of_priors = dict_of_priors

    def __call__(self, cube):
        for i, (key, prior) in enumerate(self.dict_of_priors.items()):
            cube[i] = prior(cube[i])

        if self.correlator is not None:
            cube[i] = self.correlator()
        return cube

    def sample(self, cube):
        cube = self(cube)
        return {key: cube[i] for i, key in enumerate(self._parameter_names)}


class OldPrior(BaseClass, metaclass=abc.ABCMeta):
    """
    This is the base class which can be used to include a new type of prior
    to the IMAGINE pipeline. If you want to use a distribution from
    scipy, please look at `ScipyPrior`. If you want to construct a prior from
    a sample, see `CustomPrior`.
    """

    def __init__(self, unit=None, interval=None, pdf_npoints=1500):

        if unit is None:
            unit = u.Unit(s='')

        self.unit = unit
        if interval is None:
            self._interval = [None, None]
        else:
            self._interval = interval
        self._pdf_npoints = pdf_npoints
        # Placeholders
        self._cdf = None
        self._inv_cdf = None
        self._distr = None
        self._pdf = None
        self.samples = None

    def pdf(self, x):
        """
        Probability density function (PDF) associated with this prior.
        """
        if self._pdf is None:
            self._pdf = interp1d(x=self._pdf_x, y=self._pdf_y)

        unit, [x_val] = unit_checker(self.unit, [x])

        return self._pdf(x)

    @property
    def cdf(self):
        """
        Cumulative distribution function (CDF) associated with this prior.
        """
        if self._cdf is None:
            xmin_val = self.range[0].value
            dx = (self.range[1].value-self.range[0].value)/self._pdf_npoints
            cdf_y = np.append(0, np.cumsum(self._pdf_y * dx))
            cdf_y /= cdf_y.max()  # Avoids potential problems due to truncation
            cdf_x = np.append(xmin_val, self._pdf_x + dx)
            self._cdf = interp1d(cdf_x, cdf_y)
        return self._cdf

    @property
    def inv_cdf(self):
        if self._inv_cdf is None:
            xmin_val = self.range[0].value
            dx = (self.range[1].value-self.range[0].value)/self._pdf_npoints
            cdf_y = np.append(0, np.cumsum(self._pdf_y * dx))
            cdf_y /= cdf_y.max()  # Avoids potential problems due to truncation
            cdf_x = np.append(xmin_val, self._pdf_x + dx)
            self._inv_cdf = interp1d(cdf_y, cdf_x)

        return self._inv_cdf

    def __call__(self, x):
        """
        The "prior mapping", i.e. returns the value of the
        inverse of the CDF at point(s) `x`.
        This only returns a fair sample of the prior if x is uniformly distributed.
        """
        return self.inv_cdf(x) << self.unit

    @property
    def distr(self, distr):
        """
        Constructs a scipy distribution based on an IMAGINE prior
        """

        if distr is None:
            pdf = self.pdf
            cdf = self.cdf
            ppf = self.inv_cdf

            class Distr(stats.rv_continuous):

                def _pdf(self, x):
                    return pdf(x)

                def _cdf(self, x):
                    return cdf(x)

                def _ppf(self, x):
                    return ppf(x)

        return Distr(a=self.interval[0], b=self.interval[1])


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
