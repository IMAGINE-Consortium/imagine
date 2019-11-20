"""
ensemble likelihood, described in IMAGINE techincal report
in principle
it combines covariance matrices from both observations and simulations
"""
import numpy as np
import logging as log
from copy import deepcopy
from imagine.observables.observable_dict import Simulations
from imagine.likelihoods.likelihood import Likelihood
from imagine.tools.covariance_estimator import oas_mcov
from imagine.tools.mpi_helper import mpi_slogdet, mpi_lu_solve, mpi_trace
from imagine.tools.icy_decorator import icy


@icy
class EnsembleLikelihood(Likelihood):

    def __init__(self, measurement_dict, covariance_dict=None, mask_dict=None):
        """
        EnsembleLikelihood class initialization function
        
        parameters
        ----------
        
        measurement_dict
            Measurements object
            
        covariance_dict
            Covariances object
        
        mask_dict
            Masks object
        """
        log.debug('@ ensemble_likelihood::__init__')
        super(EnsembleLikelihood, self).__init__(measurement_dict, covariance_dict, mask_dict)
    
    def __call__(self, observable_dict):
        """
        EnsembleLikelihood class call function
        
        parameters
        ----------
        
        observable_dict
            Simulations object
            
        return
        ------
        log-likelihood value (copied to all nodes)
        """
        log.debug('@ ensemble_likelihood::__call__')
        assert isinstance(observable_dict, Simulations)
        # check dict entries
        assert (observable_dict.keys() == self._measurement_dict.keys())
        likelicache = float(0)
        if self._covariance_dict is None:
            for name in self._measurement_dict.keys():
                obs_mean, obs_cov = oas_mcov(observable_dict[name].data)  # to distributed data
                data = deepcopy(self._measurement_dict[name].data)  # to distributed data
                diff = np.nan_to_num(data - obs_mean)
                if (mpi_trace(obs_cov) < 1E-28):  # zero will not be reached, at most E-32
                    likelicache += -0.5*np.vdot(diff, diff)
                else:
                    sign, logdet = mpi_slogdet(obs_cov*2.*np.pi)
                    likelicache += -0.5*(np.vdot(diff, mpi_lu_solve(obs_cov, diff))+sign*logdet)
        else:
            for name in self._measurement_dict.keys():
                obs_mean, obs_cov = oas_mcov(observable_dict[name].data)  # to distributed data
                data = deepcopy(self._measurement_dict[name].data)  # to distributed data
                diff = np.nan_to_num(data - obs_mean)
                if name in self._covariance_dict.keys():  # not all measurements have cov
                    full_cov = deepcopy(self._covariance_dict[name].data) + obs_cov
                    if (mpi_trace(full_cov) < 1E-28):  # zero will not be reached, at most E-32
                        likelicache += -0.5*np.vdot(diff, diff)
                    else:
                        sign, logdet = mpi_slogdet(full_cov*2.*np.pi)
                        likelicache += -0.5*(np.vdot(diff, mpi_lu_solve(full_cov, diff))+sign*logdet)
                else:
                    if (mpi_trace(obs_cov) < 1E-28):  # zero will not be reached, at most E-32
                        likelicache += -0.5*np.vdot(diff, diff)
                    else:
                        sign, logdet = mpi_slogdet(obs_cov*2.*np.pi)
                        likelicache += -0.5*(np.vdot(diff, mpi_lu_solve(obs_cov, diff))+sign*logdet)
        return likelicache
