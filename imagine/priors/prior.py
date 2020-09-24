# %% IMPORTS
# Built-in imports
import abc
import logging as log

# Package imports
import astropy.units as u
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d

# IMAGINE imports
from imagine.tools import BaseClass, req_attr

# All declaration
__all__ = ['Prior', 'ScipyPrior', 'CustomPrior']


# %% CLASS DEFINITIONS
class Prior(BaseClass, metaclass=abc.ABCMeta):
    """
    This is the base class which can be used to include a new type of prior
    to the IMAGINE pipeline. If you are willing to use a distribution from
    scipy, please look at `ScipyPrior`. If you want to construct a prior from
    a sample, see `CustomPrior`.
    """
    def __init__(self, xmin=None, xmax=None, unit=None, pdf_npoints=1500):

        # Ensures interval is quantity with consistent units
        unit, [xmin_val, xmax_val] = self.unit_checker(unit, [xmin, xmax])

        if unit is None:
            unit = u.Unit(s='')

        self.unit = unit
        if xmin_val is None:
            xmin_val = -np.inf
        if xmax_val is None:
            xmax_val = np.inf
        self.range = [xmin_val, xmax_val]*unit
        self._pdf_npoints = pdf_npoints
        # Placeholders
        self._cdf = None
        self._inv_cdf = None
        self._distr = None
        self._pdf = None
        self.samples = None

    @staticmethod
    def unit_checker(unit, list_of_quant):
        ul = []
        for uq in list_of_quant:
            if isinstance(uq, u.Quantity):
                if unit is None:
                    unit = uq.unit
                else:
                    uq.to(unit)
                ul.append(uq.to_value(unit))
            else:
                ul.append(uq)
        return unit, ul

    def pdf(self, x):
        """
        Probability density function (PDF) associated with this prior.
        """
        if self._pdf is None:
            self._pdf = interp1d(x=self._pdf_x, y=self._pdf_y)

        unit, [x_val] = self.unit_checker(self.unit, [x])

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
            cdf_y /= cdf_y.max() # Avoids potential problems due to truncation
            cdf_x = np.append(xmin_val, self._pdf_x + dx)
            self._cdf = interp1d(cdf_x, cdf_y)
        return self._cdf

    @property
    def inv_cdf(self):
        if self._inv_cdf is None:
            xmin_val = self.range[0].value
            dx = (self.range[1].value-self.range[0].value)/self._pdf_npoints
            cdf_y = np.append(0, np.cumsum(self._pdf_y * dx))
            cdf_y /= cdf_y.max() # Avoids potential problems due to truncation
            cdf_x = np.append(xmin_val, self._pdf_x + dx)
            self._inv_cdf = interp1d(cdf_y, cdf_x)

        return self._inv_cdf

    def __call__(self, x):
        """
        The "prior mapping", i.e. returns the value of the
        inverse of the CDF at point(s) `x`.
        """
        return self.inv_cdf(x) << self.unit

    @property
    def scipy_distr(self):
        """
        Constructs a scipy distribution based on an IMAGINE prior
        """

        if self._distr is None:
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

            if interval is None:
                interval = [None, None]
            self._distr = Distr(a=interval[0], b=interval[1])
        return self._distr



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
    xmin, xmax : float
        A pair of points representing, respectively, the minimum/maximum
        parameter values to be considered. If not provided (or set to `None`),
        the smallest/largest value in the sample minus/plus one standard
        deviation will be used.
    unit : astropy.units.Unit
        If present, sets the units used for this parameter. If absent, this
        is inferred from `xmin` and `xmax`.
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
                 unit=None, bw_method=None, pdf_npoints=1500, samples_ref=True):
        # If needed, constructs a pdf function from samples, using KDE
        if samples is not None:
            unit, [xmin_val, xmax_val, samples_val] = self.unit_checker(unit, [xmin, xmax, samples])
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
            unit, [xmin_val, xmax_val] = self.unit_checker(unit, [xmin, xmax])

        # Evaluates the PDF
        pdf_x = np.linspace(xmin_val, xmax_val, pdf_npoints)
        pdf_y = pdf_fun(pdf_x)

        super().__init__(xmin=xmin_val, xmax=xmax_val, unit=unit,
                         pdf_npoints=pdf_npoints)
        dx = (xmax_val - xmin_val) / pdf_npoints
        inv_norm = pdf_y.sum() * dx
        pdf_y = (pdf_y / inv_norm)

        self._pdf_y = pdf_y
        self._pdf_x = pdf_x

        if (samples is not None) and samples_ref:
            # If requested, stores the samples to allow later (in the Pipeline)
            # determination of correlations between parameters
            self.samples = samples << self.unit  # Adjusts units to avoid errors


class ScipyPrior(Prior):
    """
    Constructs a prior from a continuous distribution defined in
    `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`_.

    Parameters
    -----------
    distr : scipy.stats.rv_continuous
        A distribution function expressed as an instance of
        :py:class:`scipy.stats.rv_continuous`.
    *args :
        Any positional arguments required by the function selected in
        :py:data:`distr` (e.g for `scipy.
        .chi2 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html>`_, one needs to
        supply the number of degrees of freedom, :py:data:`df`)
    loc : float
        Same meaning as in :py:class:`scipy.stats.rv_continuous`: sets the
        centre of the distribution (generally, the mean or mode).
    scale : float
        Same meaning as in :py:class:`scipy.stats.rv_continuous`: sets the
        width of the distribution (e.g. the standard deviation in the normal
        case).
    xmin, xmax : float
        A pair of points representing, respectively, the minimum/maximum
        parameter values to be considered (note that this will truncate the
        original scipy distribution provided).
        If these are not provided (or set to `None`), the prior range is
        assumed to run from -infinity to infinity (in this case, `unit`
        *must* be provided).
    unit : astropy.units.Unit
        If present, sets the units used for this parameter. If absent, this
        is inferred from `xmin` and `xmax`.
    """
    def __init__(self, distr, *args, loc=0.0, scale=1.0, xmin=None, xmax=None,
                 unit=None, pdf_npoints=1500, **kwargs):
        super().__init__(xmin=xmin, xmax=xmax, unit=unit,
                         pdf_npoints=pdf_npoints)

        assert isinstance(distr, stats.rv_continuous), 'distr must be instance of scipy.stats.rv_continuous'

        distr_instance = distr(*args, loc=loc, scale=scale, **kwargs)

        if (xmin is None) and (xmax is None):
            self._pdf = distr_instance.pdf
            self._cdf = distr_instance.cdf
            self._inv_cdf = distr_instance.ppf
            self._distr = distr_instance
        else:
            # If a trucated distribution is required, proceed as in
            # the empirical case
            unit, [xmin_val, xmax_val] = self.unit_checker(unit, [xmin, xmax])
            self._pdf_x = np.linspace(xmin_val, xmax_val, pdf_npoints)
            self._pdf_y = distr_instance.pdf(self._pdf_x)
