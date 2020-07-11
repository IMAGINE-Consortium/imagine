# %% IMPORTS
# Built-in imports
from copy import deepcopy
import logging as log

# Package imports
import numpy as np

# IMAGINE imports
from imagine.likelihoods import Likelihood
from imagine.observables.observable_dict import Simulations
from imagine.tools.parallel_ops import pslogdet, plu_solve

# All declaration
__all__ = ['SimpleLikelihood']


# %% CLASS DEFINITIONS
class SimpleLikelihood(Likelihood):
    """
    A simple Likelihood class

    Parameters
    ----------
    measurement_dict : imagine.observables.observable_dict.Measurements
        Measurements
    covariance_dict : imagine.observables.observable_dict.Covariances
        Covariances
    mask_dict : imagine.observables.observable_dict.Masks
        Masks
    """

    def call(self, observable_dict):
        """
        SimpleLikelihood object call function

        Parameters
        ----------
        observable_dict : imagine.observables.observable_dict.Simulations
            Simulations object

        Returns
        ------
        likelicache : float
            log-likelihood value (copied to all nodes)
        """
        log.debug('@ simple_likelihood::__call__')
        assert isinstance(observable_dict, Simulations)
        # check dict entries
        assert (observable_dict.keys() == self._measurement_dict.keys())
        likelicache = 0
        if self._covariance_dict is None:  # no covariance matrix
            for name in self._measurement_dict.keys():
                obs_mean = deepcopy(observable_dict[name].ensemble_mean)  # use mpi_mean, copied to all nodes
                data = deepcopy(self._measurement_dict[name].data)  # to distributed data
                diff = np.nan_to_num(data - obs_mean)
                likelicache += -0.5*np.vdot(diff, diff)  # copied to all nodes
        else:  # with covariance matrix
            for name in self._measurement_dict.keys():
                obs_mean = deepcopy(observable_dict[name].ensemble_mean)  # use mpi_mean, copied to all nodes
                data = deepcopy(self._measurement_dict[name].data)  # to distributed data
                diff = np.nan_to_num(data - obs_mean)
                if name in self._covariance_dict.keys():  # not all measreuments have cov
                    cov = deepcopy(self._covariance_dict[name].data)  # to distributed data
                    sign, logdet = pslogdet(cov*2*np.pi)
                    likelicache += -0.5*(np.vdot(diff, plu_solve(cov, diff))+sign*logdet)
                else:
                    likelicache += -0.5*np.vdot(diff, diff)
        return likelicache
