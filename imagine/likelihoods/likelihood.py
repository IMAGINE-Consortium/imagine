"""
Likelihood class defines likelihood posterior function
to be used in Bayesian analysis

member fuctions:

__init__

    requires
    Measurements object
    Covariances object (optional)
    Masks object (optional)

call

    running LOG-likelihood calculation requires
    ObservableDict object
"""

# %% IMPORTS
# Built-in imports
import abc

import astropy.units as u

# Package imports
import numpy as np

# IMAGINE imports
from imagine.observables.observable_dict import (
    Measurements, Covariances, Masks)
from imagine.tools import BaseClass
from imagine.tools.parallel_ops import pslogdet, plu_solve

from ..fields.TOGOModel import Model
from ..grid.TOGOGrid import ParameterSpace, ScalarSpace

# All declaration
__all__ = ['Likelihood']


# %% CLASS DEFINITIONS

class TOGOLogLikelihood(Model):
    def __init__(self, data, simulator=None, name=None,  call_by_method=False ):
        if not isinstance(data, np.ndarray):
            raise TypeError()
        self._data = data
        name = 'Likelihood' if name is None else name

        if simulator is None:
            input_param_space = ParameterSpace(len(data), name)

            def sim(par):
                return par
            self._simulate = sim
        else:
            input_param_space = simulator.output_param_space
            self._simulate = simulator.compute_model

        super().__init__( input_param_space, ScalarSpace('Energy'), call_by_method)

    @property
    def data(self):
        return self._data


class TOGOGaussianLogLikelihood(TOGOLogLikelihood):
    def __init__(self, data, noise_cov, simulator=None, name=None):
        super().__init__(data, simulator, name, True)

        if isinstance(data, u.Quantity):
            unit = data.unit
            if not isinstance(noise_cov, u.Quantity) and unit != u.Unit(''):
                raise TypeError()
            if not noise_cov.unit == unit**2:
                raise TypeError()
        else:
            unit = u.Unit('')
            if isinstance(noise_cov, u.Quantity):
                if not unit == noise_cov.unit:
                    raise TypeError()
        if simulator is not None:
            if not unit == simulator.unit:
                raise TypeError()

        self._noise_cov = noise_cov
        if isinstance(noise_cov.value, (int, float)):
            if noise_cov <= 0:
                raise ValueError('Imagine.GaussianLogLikelihood: scalar noise_cov must be positive')
            logdet = len(self.data)*np.log(noise_cov.value*2.*np.pi)

            def apply(vec):
                return -0.5*np.vdot(vec, vec)/noise_cov - logdet
            self._apply_noise_cov = apply

        elif len(noise_cov.shape) == 1:
            # Diagonal
            if not (noise_cov >= 0).all():
                raise ValueError('Imagine.GaussianLogLikelihood: diagonal noise_cov must be positive definite')
            logdet = np.log(noise_cov*2.*np.pi).sum()

            def apply(vec):
                return -0.5*np.vdot(vec, 1./noise_cov * vec) - logdet
            self._apply_noise_cov = apply

        elif len(noise_cov.shape) == 2:
            # Full matrix
            if not (noise_cov == noise_cov.T).all():
                raise ValueError('Imagine.GaussianLogLikelihood: inverse_noise_cov must be symmetric')

            sign, logdet = pslogdet(noise_cov*2*np.pi)
            try:
                np.linalg.cholesky(noise_cov)
            except np.LinAlgError:
                print('Imagine.GaussianLogLikelihood: noise_cov not positive definite')

            def apply(vec):
                return -0.5*np.vdot(vec, plu_solve(noise_cov, vec)) - 0.5*sign*logdet

            self._apply_noise_cov = apply

        else:
            raise TypeError('Imagine.GaussianLogLikelihood: noise_cov must be either a scalar or one or two D array')

    @property
    def inverse_noise_cov(self):
        return self._inverse_noise_cov

    def compute_model(self, parameters):
        residual = self.data - self._simulate(parameters)
        return self._apply_noise_cov(residual)


class Likelihood(BaseClass, metaclass=abc.ABCMeta):
    """
    Base class that defines likelihood posterior function
    to be used in Bayesian analysis

    Parameters
    ----------
    measurement_dict : imagine.observables.observable_dict.Measurements
        A :py:obj:`Measurements <imagine.observables.observable_dict.Measurements>`
        dictionary containing observational data.
    covariance_dict : imagine.observables.observable_dict.Covariances
        A :py:obj:`Covariances <imagine.observables.observable_dict.Covariances>`
        dictionary containing observed covariance data.
        If set to `None` (the usual case), the :py:obj:`Likelihood` will try
        to find the :py:obj:`Covariances <imagine.observables.observable_dict.Covariances>`
        in the :py:data:`cov` attribute of the supplied `measurement_dict`.
    mask_dict : imagine.observables.observable_dict.Masks
        A :py:obj:`Masks <imagine.observables.observable_dict.Masks>` dictionary
        which should be applied to the measured and simulated data.
    compute_dispersion : bool
        If True, calling the Likelihood object will return the likelihood value
        and the dispersion estimated by bootstrapping the simulations object
        and computing the sample standard deviation.
        If False (default), only the likelihood value is returned.
    n_bootstrap : int
        Number of resamples used in the bootstrapping of the simulations if
        compute_dispersion is set to `True`.
    """
    def __init__(self, measurement_dict, covariance_dict=None, mask_dict=None,
                 compute_dispersion=False, n_bootstrap=150):
        # Call super constructor
        super().__init__()

        self._check_units(measurement_dict, covariance_dict)
        self.mask_dict = mask_dict
        self.measurement_dict = measurement_dict
        if covariance_dict is None:
            covariance_dict = measurement_dict.cov
        self.covariance_dict = covariance_dict
        self.compute_dispersion = compute_dispersion
        self.n_bootstrap = n_bootstrap

    def __call__(self, observable_dict, **kwargs):
        if self.mask_dict is not None:
            observable_dict = self.mask_dict(observable_dict)

        likelihood = self.call(observable_dict, **kwargs)

        if not self.compute_dispersion:
            return likelihood
        else:
            bootstrap_sample = [self._bootstrapped_likelihood(observable_dict,
                                                              **kwargs)
                                for _ in range(self.n_bootstrap)]
            dispersion = np.std(bootstrap_sample)
            return likelihood, dispersion

    def _bootstrapped_likelihood(self, simulations, **kwargs):
        # Gets ensemble size from first entry in the ObservableDict
        size, _ = simulations[list(simulations.keys())[0]].shape
        # Resamples with replacement
        idx = np.random.randint(0, size, size)
        sims_new = simulations.sub_sim(idx)
        return self.call(sims_new, **kwargs)

    @property
    def mask_dict(self):
        """
        :py:obj:`Masks <imagine.observables.observable_dict.Masks>` dictionary associated with
        this object
        """
        return self._mask_dict

    @mask_dict.setter
    def mask_dict(self, mask_dict):
        if mask_dict is not None:
            assert isinstance(mask_dict, Masks)
        self._mask_dict = mask_dict

    @property
    def measurement_dict(self):
        """
        :py:obj:`Measurements <imagine.observables.observable_dict.Measurements>` dictionary associated with
        this object

        NB If a mask is used, only the masked version is stored
        """
        return self._measurement_dict

    @measurement_dict.setter
    def measurement_dict(self, measurement_dict):
        assert isinstance(measurement_dict, Measurements)
        self._measurement_dict = measurement_dict
        if self._mask_dict is not None:  # apply mask
            self._measurement_dict = self.mask_dict(self._measurement_dict)

    @property
    def covariance_dict(self):
        """
        :py:obj:`Covariances <imagine.observables.observable_dict.Covariances>` dictionary associated with
        this object

        NB If a mask is used, only the masked version is stored
        """
        return self._covariance_dict

    @covariance_dict.setter
    def covariance_dict(self, covariance_dict):
        if covariance_dict is not None:
            assert isinstance(covariance_dict, Covariances)
        self._covariance_dict = covariance_dict
        if (self._mask_dict is not None) and (self._covariance_dict is not None):
            self._covariance_dict = self.mask_dict(self._covariance_dict)

    @abc.abstractmethod
    def call(self, observable_dict):
        """
        Parameters
        ----------
        observable_dict : imagine.observables.observable_dict
        variables
        """
        raise NotImplementedError

    @staticmethod
    def _check_units(measurements, covariances):
        """
        Makes sure that measurements and covariances units are compatible
        """
        if covariances is None:
            return
        for k in measurements:
            if measurements[k].unit is None:
                assert covariances[k].unit is None
            else:
                assert (measurements[k].unit)**2 == covariances[k].unit
