import numpy as np
import logging as log
from copy import deepcopy

from imagine.observables.observable import Observable
from imagine.observables.observable_dict import Measurements, Simulations, Covariances, Masks
from imagine.likelihoods.likelihood import Likelihood
from imagine.tools.icy_decorator import icy


@icy
class SimpleLikelihood(Likelihood):

    def __init__(self, measurement_dict, covariance_dict=None, mask_dict=None):
        """

        :param measurement_dict: Measurements object
        :param covariance_dict: Covariances object
        :param mask_dict: Masks object
        """
        super(SimpleLikelihood, self).__init__(measurement_dict, covariance_dict, mask_dict)

    def __call__(self, observable_dict, variables=None):
        """

        :param observable_dict: Simulations object
        :return: log-likelihood value
        """
        # parse variables
        work_parameters = dict()
        if variables is not None and variables != dict():
            assert (tuple(variables.keys()) == self._active_parameters)
            work_parameters.update(self._map_variables_to_parameters(variables))
        #
        assert isinstance(observable_dict, Simulations)
        # check dict entries
        assert (observable_dict.keys() == self._measurement_dict.keys())
        likelicache = float(0)
        if self._covariance_dict is None:
            for name in self._measurement_dict.keys():
                obs_mean = deepcopy(observable_dict[name].ensemble_mean)
                data = deepcopy(self._measurement_dict[name].to_global_data())
                diff = np.nan_to_num(data - obs_mean)
                likelicache += -float(0.5)*float(np.vdot(diff, diff))
        else:
            for name in self._measurement_dict.keys():
                obs_mean = deepcopy(observable_dict[name].ensemble_mean)
                data = deepcopy(self._measurement_dict[name].to_global_data())
                diff = np.nan_to_num(data - obs_mean)
                if name in self._covariance_dict.keys():  # not all measreuments have cov
                    cov = deepcopy(self._covariance_dict[name].to_global_data())
                    (sign, logdet) = np.linalg.slogdet(cov*2.*np.pi)
                    likelicache += -float(0.5)*float(np.vdot(diff, np.linalg.solve(cov, diff.T))+sign*logdet)
                else:
                    likelicache += -float(0.5)*float(np.vdot(diff, diff))
        return likelicache
