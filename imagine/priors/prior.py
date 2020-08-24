# %% IMPORTS
# Package imports
import astropy.units as u
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import matplotlib.pylab as pl

# All declaration
__all__ = ['GeneralPrior', 'ScipyPrior', 'EmpiricalPrior']


# %% CLASS DEFINITIONS
class GeneralPrior:
    """

    """
    def __init__(self, xmin=None, xmax=None, unit=None):

        # Ensures interval is quantity with consistent units
        unit, xm_l = self.unit_checker(unit, [xmin, xmax])

        if unit is None:
            unit = u.Unit(s='')

        self.unit = unit
        self.range = [xmin, xmax]*unit

        # Placeholders
        self._cdf = None
        self._inv_cdf = None
        self.distr = None
        self._pdf = None

    @staticmethod
    def unit_checker(unit, list_of_quant):
        ul = []
        for uq in list_of_quant:
            if isinstance(uq, u.Quantity):
                if unit is None:
                    unit = uq.unit
                else:
                    uq.to(unit)
                ul.append(uq.value)
            else:
                ul.append(uq)
        return unit, ul

    @property
    def pdf(self):
        """
        Probability density function (PDF) associated with this prior.
        """
#        return self.distr.pdf
        return self._pdf
#    def pdf_unscaled(self, x):
#        """
#        Probability density function (PDF) associated with this prior.
#        """
#        return self.pdf(self._scale_parameter(x))

#    def _scale_parameter(self, x):
#        return (x - self.range[0])/(self.range[1] - self.range[0])

    @property
    def cdf(self):
        """
        Cumulative distribution function (CDF) associated with this prior.
        """
        return self._cdf
        # return self.distr.cdf

    @property
    def inv_cdf(self):
        return self._inv_cdf

        #return self.distr.ppf

    def __call__(self, x):
        """
        The "prior mapping", i.e. returns the value of the
        inverse of the CDF at point(s) `x`.
        """
        return self.inv_cdf(x) #*self.unit



    @staticmethod
    def distr_initilalizer(pdf=None, cdf=None, ppf=None, interval=None):
        if all([pdf, cdf, ppf]) is None:
            raise ValueError('At least a cdf, a pdf or a ppf must be specified!')

        class Distr(stats.rv_continuous):

            def _pdf(self, x):
                return pdf(x)
            def _cdf(self, x):
                return cdf(x)
            def _ppf(self, x):
                return ppf(x)

        if interval is None:
            interval = [None, None]
        return Distr(a=interval[0], b=interval[1])


# %% CLASS DEFINITIONS
class EmpiricalPrior(GeneralPrior):
    """
    Allows constructing a prior from a pre-existing sampling of the parameter
    space or a known probability density function (PDF).

    Like in `MultiNest <https://johannesbuchner.github.io/PyMultiNest/pymultinest_run.html>`_, priors here are represented by a mapping of uniformly
    distributed scaled parameter values into a distribution with the
    desired properties (i.e. following the expected PDF). Such mapping is
    equivalent to the inverse of the cumulative distribution function (CDF)
    associated with the prior distribution.


    Notes
    -----
    Here we make a summary of the algorithm that allows finding the inverse of
    the CDF (and therefore, a IMAGINE prior) from a set of samples or PDF.
    (1) If set of samples is provided, computes Kernel Density Estimate (KDE)
    representation of it using Gaussian kernels.
    (2) The KDE is evaluated on `pdf_npoints` and thi si used to construct
    a interpolated cubic spline, which can be inspected through the method
    :py:meth:`pdf`.
    (4) From PDF spline, the CDF is computed, which can be accessed using
    :py:meth:`cdf`.
    (5) The CDF is evaluated on `inv_cdf_npoints`, and the inverse of the CDF
    is, again, constructed as a interpolated cubic spline. The spline object
    is available at :py:meth:`inv_cdf`.


    Parameters
    ----------
    samples : array_like
        Array containing a sample of the prior distribution one wants to use.
        Note: this will use :py:class:`scipy.stats.gaussian_kde` to compute
        the probability density function (PDF) through kernel density
        estimate using Gaussian kernels.
    pdf_fun : function
        A Python function containing the PDF associated with this prior.
        If an no `interval` is provided, the domain of PDF(x) will be assumed to be
        within the interval [0,1].
    pdf_x, pdf_y : array_like
        The PDF can be provided as two arrays of points following
        (pdf_x, pdf_y) = (x, PDF(x)). In the absence of an `interval`, the
        interval [min(pdf_x), max(pdf_x)] will be used.
    interval : tuple or list
        A pair of points representing, respectively, the minimum and maximum
        parameter values to be considered.
    bw_method: scalar or str
        Used by :py:class:`scipy.stats.gaussian_kde` to select the bandwidth
        employed to estimate the PDF from provided samples. Can be a number,
        if using fixed bandwidth, or strings ‘scott’ or ‘silverman’ if these
        rules are to be selected.
    pdf_npoints : int
        Number of points used to evaluate pdf_fun or the KDE constructed from
        the samples.
    inv_cdf_npoints : int
        Number of points used to evaluate the CDF for the calculation of its
        inverse.
    """
    def __init__(self, samples=None, pdf_fun=None,
                 pdf_x=None, pdf_y=None, xmin=None, xmax=None,
                 bw_method=None, pdf_npoints=1500, unit=None):
        if pdf_x is None:
            if pdf_y is None:
                # If needed, constructs a pdf function from samples, using KDE
                if samples is not None:
                    assert pdf_fun is None, 'Either provide the samples or the PDF, not both.'
                    unit, li = self.unit_checker(unit, [xmin, xmax, samples])
                    assert unit is not None, 'At least one input must have a unit or a astropy unit must be provided'
                    if xmin or xmax is None:
                        std = np.std(li[2])
                    if xmin is None:
                        li[0] = li[2].min() - std
                    if xmax is None:
                        li[1] = li[2].max() + std
                    self.samples = li[2]*unit

                    pdf_fun = stats.gaussian_kde(li[2], bw_method=bw_method)
                else:
                    assert (xmin is not None and xmax is not None), 'both xmin and xmax must be given for pdf_fun'
                    unit, li = self.unit_checker(unit, [xmin, xmax])
                if pdf_fun is not None:
                    # Evaluates the PDF
                    pdf_x = np.linspace(li[0], li[1], pdf_npoints)
                    pdf_y = pdf_fun(pdf_x)
                else:
                    raise ValueError('either samples or pdf_fun or pdf_y must be given')

        else:
            li = [pdf_x.min(), pdf_x.max()]
        super().__init__(xmin=li[0], xmax=li[1], unit=unit)
        dx = (li[1] - li[0]) / pdf_npoints
        inv_norm = pdf_y.sum() * dx
        pdf_y = (pdf_y / inv_norm)
        self._pdf = interp1d(x=pdf_x, y=pdf_y)
        self._cdf = interp1d(np.append(li[0], pdf_x + dx), np.append(0, np.cumsum(pdf_y * dx)))
        self._inv_cdf = interp1d(np.append(0, np.cumsum(pdf_y * dx)), np.append(li[0], pdf_x + dx))

    @property
    def pdf(self):
        """
        Probability density function (PDF) associated with this prior.
        """
        return self._pdf

    @property
    def cdf(self):
        """
        Cumulative distribution function (CDF) associated with this prior.
        """
        return self._cdf

    @property
    def inv_cdf(self):
        """
        Inverse of the CDF associated with this prior,
        """
        return self._inv_cdf


class ScipyPrior(GeneralPrior):
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
        :py:data:`distr` (e.g for `scipy.stats.chi2 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html>`_, one needs to
        supply the number of degrees of freedom, :py:data:`df`)
    loc : float
        Same meaning as in :py:class:`scipy.stats.rv_continuous`: sets the
        centre of the distribution (generally, the mean or mode).
    scale : float
        Same meaning as in :py:class:`scipy.stats.rv_continuous`: sets the
        width of the distribution (e.g. the standard deviation in the normal
        case).
    interval : tuple or list
        A pair of points representing, respectively, the minimum and maximum
        parameter values to be considered (will be used to rescale the
        interval).
    """
    def __init__(self, distr, unit=None, *args, loc=0.0, scale=1.0, xmin=None, xmax=None, **kwargs):
        super().__init__(xmin=xmin, xmax=xmax, unit=unit)
        assert self.unit is not None, 'Either the interval has units or a unit must be provided'

        assert isinstance(distr, stats.rv_continuous), 'distr must be instance of scipy.stats.rv_continuous'

        self.distr = distr(*args, loc=loc, scale=scale, **kwargs)

        self._pdf = self.distr.pdf
        self._cdf = self.distr.cdf

        # In principle, it should be possible to use scipy's built-in
        # ppf for this (which is supposed to be accurate and fast),
        # but this has not yet being implemented as it requires
        # some extra rescaling to work with the truncation
        self._inv_cdf = self.distr.ppf
