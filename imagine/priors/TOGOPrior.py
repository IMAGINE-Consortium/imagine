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
__all__ = ['UnivariatePrior', 'MultivariatePrior']


class UnivariatePrior(Model):
    def __init__(self, pd_info, parameter_name, unit=None, pdf_n_points=1000, numerical_range=None, **distr_kwargs):
        if unit is None:
            unit = u.Unit(s='')
        param_space = ScalarSpace(parameter_name)
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
            self._distr = f(**distr_kwargs)
        elif isinstance(f, str):
            self._distr = getattr(stats, f)(**distr_kwargs)

        elif isinstance(f, (collections.Callable, np.ndarray,)):
            if isinstance(f, np.nd_array):
                if len(f.shape) != 1:
                    raise ValueError('Imagine.UnivariatePrior: Samples must be provided as a 1d array')
                f = stats.gaussian_kde(f, bw_method='scott')
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
            raise TypeError('Imagine.UnivariatePrior must me set with either a <scipy.stats.rv_continous> object, ' +
                            'a string indicating a scipy distr object or a callable parametrizing the pdf')

# %% CLASS DEFINITIONS


class MultivariatePrior(Model):

    def __init__(self, pd_info, correlator=None, unit=None,  **distr_kwargs):

        if correlator is not None:
            print('Imagine.MultivariatePrior: correlator feature not yet implemented, the provided instance has no effect')
        self.correlator = correlator
        if isinstance(pd_info, dict):
            self._unit = unit
            param_space = ParameterSpaceDict({n: prior.output_param_space for n, prior in pd_info.items()})

            def draw():
                return {n: prior.draw_sample() for n, prior in pd_info.items()}

            def compute(self, parameters):
                return {n: prior(parameters[n]) for n, prior in pd_info.items()}

            self._draw = draw
            self._compute = compute

        else:
            if unit is None:
                self._unit = u.Unit('')
            else:
                self._unit = unit

            name = distr_kwargs.pop('name', 'Prior')

            n = distr_kwargs.pop('n', None)
            if n is None:
                try:
                    n = len(distr_kwargs['loc'])
                except KeyError:
                    raise KeyError('Imagine.MultivariatePrior: Number of dimensions not provided either via thh key word argument <n> or via loc keyword')

            if isinstance(pd_info,  str):
                if isinstance(pd_info, str):
                    pd_info = getattr(stats, pd_info)(**distr_kwargs)

            if isinstance(pd_info, stats.rv_continuous):
                raise TypeError('Imagine.MultivariatePrior: The provided distribution is univariate')
            if not hasattr(pd_info, 'rvs'):
                raise AttributeError('Imagine.MultivariatePrior: The object provided seems to be a probability distribution')

            param_space = ParameterSpace(n, name)

            def draw():
                return pd_info.rvs() << unit

            def compute(self, parameters):
                raise NotImplementedError('Imagine.MultivariatePrior: This distribution was instantiated via a scipy distribution, which does not support ppf evaluation')

            self._draw = draw
            self._compute = compute

        super().__init__(param_space, param_space, call_by_method=True)

    def draw_sample(self):
        return self._draw()

    def compute_model(self, parameters):
        return self._compute(parameters)
