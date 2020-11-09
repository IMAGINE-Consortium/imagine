"""
ensemble likelihood, described in IMAGINE techincal report
in principle
it combines covariance matrices from both observations and simulations
"""

# %% IMPORTS
# Built-in imports
from copy import deepcopy
import logging as log

# Package imports
import numpy as np

# IMAGINE imports
from imagine.likelihoods import Likelihood
from imagine.observables.observable_dict import Simulations
from imagine.tools.covariance_estimator import oas_mcov
from imagine.tools.parallel_ops import pslogdet, plu_solve, ptrace

# All declaration
__all__ = ['EnsembleLikelihood']


# %% CLASS DEFINITIONS
class EnsembleLikelihood(Likelihood):
    """
    EnsembleLikelihood class initialization function

    Parameters
    ----------
    measurement_dict : imagine.observables.observable_dict.Measurements
        Measurements
    covariance_dict : imagine.observables.observable_dict.Covariances
        Covariances
    mask_dict : imagine.observables.observable_dict.Masks
        Masks
    """
    def __init__(self, measurement_dict, covariance_dict=None, mask_dict=None,
                 cov_func=None):

        super().__init__(measurement_dict, covariance_dict=covariance_dict,
                         mask_dict=mask_dict)

        # Requires covariaces to be present when using this type of Likelihood
        assert self._covariance_dict is not None

        if cov_func is None:
            self.cov_func = oas_mcov
        else:
            self.cov_func = cov_func

    def call(self, simulations_dict):
        """
        EnsembleLikelihood class call function

        Parameters
        ----------
        simulations_dict : imagine.observables.observable_dict.Simulations
            Simulations object

        Returns
        ------
        likelicache : float
            log-likelihood value (copied to all nodes)
        """
        log.debug('@ ensemble_likelihood::__call__')
        assert isinstance(simulations_dict, Simulations)
        assert  set(simulations_dict.keys()).issubset(self._measurement_dict.keys())
        assert  set(simulations_dict.keys()).issubset(self._covariance_dict.keys())

        likelicache = 0
        for name in simulations_dict.keys():
            # Estimated Galactic Covariance
            sim_mean, sim_cov = self.cov_func(simulations_dict[name].data)

            meas_data, meas_cov = (self._measurement_dict[name].data,
                                   self._covariance_dict[name].data)
            diff = meas_data - sim_mean

            full_cov = meas_cov + sim_cov
            sign, logdet = pslogdet(full_cov*2.*np.pi)
            likelicache += -0.5*(np.vdot(diff, plu_solve(full_cov, diff)) + sign*logdet)

        return likelicache


