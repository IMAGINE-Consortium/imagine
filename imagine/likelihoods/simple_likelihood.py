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

    def call(self, simulations_dict):
        """
        SimpleLikelihood object call function

        Parameters
        ----------
        simulations_dict : imagine.observables.observable_dict.Simulations
            Simulations object

        Returns
        ------
        likelicache : float
            log-likelihood value (copied to all nodes)
        """
        log.debug('@ simple_likelihood::__call__')
        assert isinstance(simulations_dict, Simulations)
        # check dict entries
        assert  set(simulations_dict.keys()).issubset(self._measurement_dict.keys())

        if self.covariance_dict is not None:
            covariance_dict = self._covariance_dict
        else:
            covariance_dict = {}

        likelicache = 0
        for name in self._measurement_dict:
            obs_mean = deepcopy(simulations_dict[name].ensemble_mean)
            data = deepcopy(self._measurement_dict[name].data)  # to distributed data
            diff = np.nan_to_num(data - obs_mean)

            if name in covariance_dict:  # some measurement may not have cov
                cov = self._covariance_dict[name].data
                sign, logdet = pslogdet(cov*2*np.pi)

                likelicache += -0.5*(np.vdot(diff, plu_solve(cov, diff)) + sign*logdet)
            else:
                likelicache += -0.5*np.vdot(diff, diff)

        return likelicache
