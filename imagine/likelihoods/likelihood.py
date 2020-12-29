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

import numpy as np

# IMAGINE imports
from imagine.observables.observable_dict import (
    Measurements, Covariances, Masks)
from imagine.tools import BaseClass

# All declaration
__all__ = ['Likelihood']


# %% CLASS DEFINITIONS
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
                 compute_dispersion=False, n_bootstrap=500):
        # Call super constructor
        super().__init__()

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
