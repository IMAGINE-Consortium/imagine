# %% IMPORTS
# Built-in imports
import logging as log

# Package imports
import numpy as np

# IMAGINE imports
from imagine.likelihoods import Likelihood
from imagine.observables.observable_dict import Simulations
from imagine.tools.covariance_estimator import oas_mcov
from imagine.tools.parallel_ops import (pslogdet, plu_solve, ptrace, pdiag,
                                        pvar, pmean)

# All declaration
__all__ = ['EnsembleLikelihood','EnsembleLikelihoodDiagonal']


# %% CLASS DEFINITIONS
class EnsembleLikelihood(Likelihood):
    """
    Computes the likelihood accounting for the effects of stochastic fields

    This is done by estimating the covariance associated the stochastic fields
    from an ensemble of simulations.

    Parameters
    ----------
    measurement_dict : imagine.observables.observable_dict.Measurements
        Measurements
    covariance_dict : imagine.observables.observable_dict.Covariances
        The covariances associated with the measurements. If the keyword
        argument is absent, the covariances will be read from the attribute
        `measurement_dict.cov`.
    mask_dict : imagine.observables.observable_dict.Masks
        Masks which will be applied to the Measurements, Covariances and
        Simulations, before computing the likelihood
    cov_func : func
        A function which takes a (Nens, Ndata) data array (potentially
        MPI distributed) and returns a tuple comprising the mean and an
        estimated covariance matrix.
        If absent, :py:func:`imagine.tools.covariance_estimator.oas_mcov` will
        be used.
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

        likelicache = 0.
        for name in simulations_dict.keys():
            # Estimated Galactic Covariance
            sim_mean, sim_cov = self.cov_func(simulations_dict[name].data)
            # Observed data/covariance
            meas_data, meas_cov = (self._measurement_dict[name].data,
                                   self._covariance_dict[name].data)

            diff = meas_data - sim_mean
            full_cov = meas_cov + sim_cov
            sign, logdet = pslogdet(full_cov*2.*np.pi)

            likelicache += -0.5*(np.vdot(diff, plu_solve(full_cov, diff)) + sign*logdet)

        return likelicache


class EnsembleLikelihoodDiagonal(Likelihood):
    """
    As `EnsembleLikelihood` but assuming that the covariance matrix is
    diagonal and well described by the sample variance. Likewise, only
    considers the diagonal of the observational covariance matrix.

    Parameters
    ----------
    measurement_dict : imagine.observables.observable_dict.Measurements
        Measurements
    covariance_dict : imagine.observables.observable_dict.Covariances
        The covariances associated with the measurements. If the keyword
        argument is absent, the covariances will be read from the attribute
        `measurement_dict.cov`.
    mask_dict : imagine.observables.observable_dict.Masks
        Masks which will be applied to the Measurements, Covariances and
        Simulations, before computing the likelihood
    """
    def __init__(self, measurement_dict, covariance_dict=None, mask_dict=None):

        super().__init__(measurement_dict, covariance_dict=covariance_dict,
                         mask_dict=mask_dict)

        # Requires covariaces to be present when using this type of Likelihood
        assert self._covariance_dict is not None


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

        likelicache = 0.
        for name in simulations_dict.keys():
            # Estimated Galactic Covariance
            sim_mean = pmean(simulations_dict[name].data)
            sim_var = pvar(simulations_dict[name].data)
            # Observed data/covariance
            meas_data = self._measurement_dict[name].data
            meas_var =  pdiag(self._covariance_dict[name].data)

            diff = meas_data - sim_mean
            full_var = meas_var + sim_var

            sign = np.sign(full_var).prod()
            logdet = np.log(full_var*2.*np.pi).sum()

            likelicache += -0.5*(np.vdot(diff, 1./full_var * diff) + sign*logdet)

        return likelicache
