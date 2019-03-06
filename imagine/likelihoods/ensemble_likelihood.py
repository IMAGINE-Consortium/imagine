"""
ensemble likelihood, described in IMAGINE techincal report
in principle
it combines covariance matrices from both observations and simulations
"""
import numpy as np
from copy import deepcopy

from imagine.observables.observable import Observable
from imagine.observables.observable_dict import Measurements, Simulations, Covariances
from imagine.likelihoods.likelihood import Likelihood
from imagine.tools.covariance_estimator import oas_mcov
from imagine.tools.icy_decorator import icy


@icy
class EnsembleLikelihood(Likelihood):

    def __init__(self, measurement_dict, covariance_dict=None):
        """

        :param measurement_dict: Measurements object
        :param covariance_dict: Covariances object
        """
        self.measurement_dict = measurement_dict
        self.covariance_dict = covariance_dict

    @property
    def measurement_dict(self):
        return self._measurement_dict

    @measurement_dict.setter
    def measurement_dict(self, measurement_dict):
        assert isinstance(measurement_dict, Measurements)
        self._measurement_dict = measurement_dict

    @property
    def covariance_dict(self):
        return self._covariance_dict

    @covariance_dict.setter
    def covariance_dict(self, covariance_dict):
        if covariance_dict is not None:
            assert isinstance(covariance_dict, Covariances)
        self._covariance_dict = covariance_dict
    
    #
    def __call__(self, observable_dict):
        """

        :param observable_dict: Simulations object
        :return: log-likelihood value
        """
        assert isinstance(observable_dict, Simulations)
        # check dict entries
        assert (observable_dict.keys() == self._measurement_dict.keys())
        likelicache = float(0)
        if self._covariance_dict is None:
            for name in self._measurement_dict.keys():
                obs_mean, obs_cov = oas_mcov(observable_dict[name])
                data = deepcopy(self._measurement_dict[name].to_global_data())
                diff = np.nan_to_num(data - obs_mean)
                if obs_cov.trace() < 1E-28:  # zero will not be reached, at most E-32
                    likelicache += -float(0.5)*float(np.vdot(diff, diff))
                else:
                    sign, logdet = np.linalg.slogdet(obs_cov*2.*np.pi)
                    likelicache += -float(0.5)*float(np.vdot(diff, np.linalg.solve(obs_cov, diff.T))+sign*logdet)
        else:
            for name in self._measurement_dict.keys():
                obs_mean, obs_cov = oas_mcov(observable_dict[name])
                data = deepcopy(self._measurement_dict[name].to_global_data())
                diff = np.nan_to_num(data - obs_mean)
                if name in self._covariance_dict.keys():  # not all measurements have cov
                    full_cov = deepcopy(self._covariance_dict[name].to_global_data()) + obs_cov
                    if full_cov.trace() < 1E-28:  # zero will not be reached, at most E-32
                        likelicache += -float(0.5)*float(np.vdot(diff, diff))
                    else:
                        sign, logdet = np.linalg.slogdet(full_cov*2.*np.pi)
                        likelicache += -float(0.5)*float(np.vdot(diff, np.linalg.solve(full_cov, diff.T))+sign*logdet)
                else:
                    if obs_cov.trace() < 1E-28:  # zero will not be reached, at most E-32
                        likelicache += -float(0.5)*float(np.vdot(diff, diff))
                    else:
                        sign, logdet = np.linalg.slogdet(obs_cov*2.*np.pi)
                        likelicache += -float(0.5)*float(np.vdot(diff, np.linalg.solve(obs_cov, diff.T))+sign*logdet)
        return likelicache
