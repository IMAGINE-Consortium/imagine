import numpy as np
import logging as log
from copy import deepcopy
from imagine.observables.observable_dict import Simulations
from imagine.likelihoods.likelihood import Likelihood
from imagine.tools.mpi_helper import mpi_slogdet, mpi_lu_solve
from imagine.tools.icy_decorator import icy


@icy
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
    def __init__(self, measurement_dict, covariance_dict=None, mask_dict=None):
        log.debug('@ simple_likelihood::__init__')
        super(SimpleLikelihood, self).__init__(measurement_dict, covariance_dict, mask_dict)

    def __call__(self, observable_dict):
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
        likelicache = np.float64(0)
        if self._covariance_dict is None:  # no covariance matrix
            for name in self._measurement_dict.keys():
                obs_mean = deepcopy(observable_dict[name].ensemble_mean)  # use mpi_mean, copied to all nodes
                data = deepcopy(self._measurement_dict[name].data)  # to distributed data
                diff = np.nan_to_num(data - obs_mean)
                likelicache += -float(0.5)*float(np.vdot(diff, diff))  # copied to all nodes
        else:  # with covariance matrix
            for name in self._measurement_dict.keys():
                obs_mean = deepcopy(observable_dict[name].ensemble_mean)  # use mpi_mean, copied to all nodes
                data = deepcopy(self._measurement_dict[name].data)  # to distributed data
                diff = np.nan_to_num(data - obs_mean)
                if name in self._covariance_dict.keys():  # not all measreuments have cov
                    cov = deepcopy(self._covariance_dict[name].data)  # to distributed data
                    (sign, logdet) = mpi_slogdet(cov*2.*np.pi)
                    likelicache += -0.5*(np.vdot(diff, mpi_lu_solve(cov, diff))+sign*logdet)
                else:
                    likelicache += -0.5*np.vdot(diff, diff)
        return likelicache
