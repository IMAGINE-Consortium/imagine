# %% IMPORTS
# Package imports
import astropy.units as u
import numpy as np
import scipy.stats as stats
from scipy.interpolate import CubicSpline

# All declaration
__all__ = ['GeneralPrior', 'ScipyPrior']


# %% CLASS DEFINITIONS
class GeneralPrior:
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
    def __init__(self,  interval=None):

        # Ensures interval is quantity with consistent units
        interval = u.Quantity(interval)
        self.range = interval

        # Placeholders
        self._cdf = None
        self._inv_cdf = None
        self.distr = None
        self._pdf = None

    @property
    def pdf(self):
        """
        Probability density function (PDF) associated with this prior.
        """
        return self._pdf

    def pdf_unscaled(self, x):
        """
        Probability density function (PDF) associated with this prior.
        """

        return self.pdf(self._scale_parameter(x))

    def _scale_parameter(self, x):
        return (x - self.range[0])/(self.range[1] - self.range[0])

    @property
    def cdf(self):
        """
        Cumulative distribution function (CDF) associated with this prior.
        """
        if self._cdf is None:
            if self.pdf is not None:
                self._cdf = self.pdf.antiderivative()
        return self._cdf

    @property
    def inv_cdf(self):
        return self._inv_cdf

    def __call__(self, x):
        """
        The "prior mapping", i.e. returns the value of the
        inverse of the CDF at point(s) `x`.
        """
        return self.inv_cdf(x)


class GridPrior(GeneralPrior):
    def __init__(self, interval, inv_cdf_npoints):
        super().__init__(interval)
        self.inv_cdf_npoints = inv_cdf_npoints
        self._initialize_inv_cdf()

    def _initialize_inv_cdf(self):
        """
        Inverse of the CDF associated with this prior,
        expressed as a :py:class:`scipy.interpolate.CubicSpline` object.
        """
        assert self.cdf is not None
        t = np.linspace(0, 1, self.inv_cdf_npoints)
        y = self.cdf(t)
        # Rescales the image of the function to [0,1]
        y = (y-y.min())/(y.max()-y.min())
        # For some distributions, there will be multiple
        # values of y=0 for the same t. The following corrects this
        # by simply removing this part of the cdf
        select_zeros = (y == 0)
        if len(select_zeros) > 1:
            y = y[~select_zeros]
            t = t[~select_zeros]
        # Creates interpolated spline
        self._inv_cdf = CubicSpline(y, t)


class AnalyticPrior(GridPrior):
     def __init__(self, pdf_fun, interval, inv_cdf_npoints):
         super().__init__(interval, inv_cdf_npoints)

         self._pdf = CubicSpline(pdf_x, pdf_y)



class NonAnalyticPrior(GridPrior):
    def __init__(self, pdf_x, pdf_y, interval, inv_cdf_npoints):
        super().__init__(interval, inv_cdf_npoints)
        self._pdf = CubicSpline(pdf_x, pdf_y)
        self.inv_cdf_npoints = inv_cdf_npoints


class EmpiricalPrior(GridPrior):

    def __init__(self, samples, interval, bw_method=None, pdf_npoints=1500, inv_cdf_npoints=1500):
            super().__init__()
    # PDF from samples mode -------------------
            self.inv_cdf_npoints = inv_cdf_npoints
            if interval is not None:
                ok = (samples > interval[0]) * (samples < interval[1])
                samples = samples[ok]
            pdf_fun = stats.gaussian_kde(samples, bw_method=bw_method)
            xmin, xmax = samples.min(), samples.max()
            if pdf_fun is not None:
                # Evaluates the PDF
                pdf_x = np.linspace(xmin, xmax, pdf_npoints)
                pdf_y = pdf_fun(pdf_x)
                # Normalizes and removes units
                inv_norm = pdf_y.sum() * (xmax - xmin) / pdf_npoints
                pdf_y = (pdf_y / inv_norm).value
                pdf_x = ((pdf_x - pdf_x.min()) / (pdf_x.max() - pdf_x.min())).value
                # Recovers units
                self.range = (xmin, xmax) * self.range.unit


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
    def __init__(self, distr, *args, loc=0.0, scale=1.0,
                 interval=None, **kwargs):
        super().__init__(interval=interval)

        assert isinstance(distr, stats.rv_continuous), 'distr must be instance of scipy.stats.rv_continuous'

        if interval is not None:
            xmin, xmax = interval
            scale /= (xmax - xmin)

        loc = self._scale_parameter(loc)
        self.distr = distr(*args, loc=loc, scale=scale, **kwargs)

        self._pdf = self.distr.pdf
        self._cdf = self.distr.cdf

        # In principle, it should be possible to use scipy's built-in
        # ppf for this (which is supposed to be accurate and fast),
        # but this has not yet being implemented as it requires
        # some extra rescaling to work with the truncation
        self._inv_cdf = self.distr.ppf
