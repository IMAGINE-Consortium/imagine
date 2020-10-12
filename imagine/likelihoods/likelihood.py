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
    """

    def __init__(self, measurement_dict, covariance_dict=None, mask_dict=None):
        # Call super constructor
        super().__init__()

        self.mask_dict = mask_dict
        self.measurement_dict = measurement_dict
        if covariance_dict is None:
            covariance_dict = measurement_dict.cov
        self.covariance_dict = covariance_dict

    def __call__(self, observable_dict, **kwargs):
        if self.mask_dict is not None:
            observable_dict = self.mask_dict(observable_dict)
        return(self.call(observable_dict, **kwargs))

    @property
    def mask_dict(self):
        return self._mask_dict

    @mask_dict.setter
    def mask_dict(self, mask_dict):
        if mask_dict is not None:
            assert isinstance(mask_dict, Masks)
        self._mask_dict = mask_dict

    @property
    def measurement_dict(self):
        return self._measurement_dict

    @measurement_dict.setter
    def measurement_dict(self, measurement_dict):
        assert isinstance(measurement_dict, Measurements)
        self._measurement_dict = measurement_dict
        if self._mask_dict is not None:  # apply mask
            self._measurement_dict = self.mask_dict(self._measurement_dict)

    @property
    def covariance_dict(self):
        return self._covariance_dict

    @covariance_dict.setter
    def covariance_dict(self, covariance_dict):
        if covariance_dict is not None:
            assert isinstance(covariance_dict, Covariances)
        self._covariance_dict = covariance_dict
        if self._mask_dict is not None:  # apply mask
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
