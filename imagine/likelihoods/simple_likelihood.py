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
        -------
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
            obs_data = simulations_dict[name].ensemble_mean
            meas_data = self._measurement_dict[name].data
            diff = meas_data - obs_data

            if name in covariance_dict:  
                if self._covariance_dict[name].dtype == 'variance':
                    # If only the variance was originally specified, 
                    # use it *without constructing the full covariance*
                    meas_var =  self._covariance_dict[name].var

                    sign = np.sign(meas_var).prod()
                    logdet = np.log(meas_var*2.*np.pi).sum()

                    likelicache += -0.5*np.vdot(diff, 1./meas_var * diff) - 0.5*sign*logdet
                else:
                    # If a full covariance matrix was originally specified, use it!
                    meas_cov = self._covariance_dict[name].data
                    
                    sign, logdet = pslogdet(meas_cov*2*np.pi)

                    likelicache += -0.5*np.vdot(diff, plu_solve(meas_cov, diff)) - 0.5*sign*logdet
            else:
                # some measurement may not have cov
                likelicache += -0.5*np.vdot(diff, diff)

        return likelicache
