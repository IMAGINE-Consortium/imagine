import numpy as np
from copy import deepcopy

from imagine.observables.observable import Observable
from imagine.likelihoods.likelihood import Likelihood

class SimpleLikelihood(Likelihood):

    '''
    observables (in __call__)
        -- observable (or, simulated data) dict
        entry names and data standards are defined in imagine/likelihoods/likelihood/likelihood.py
        content of observables dict is not a single map, but an ensemble
    observable_names
        -- list of tuple of entry names
    measurements
        -- observational data dict
        has the same naming convention as observables
        its content is a single measured map
    covariances
        -- observational covariance matrix dict
        follows the same naming convention as observables
        if under certain name there exists no covariance matrix, set it to None
    
    '''
    def __init__(self, observable_names, measurements, covariances):
        self.observable_names = observable_names
        # observable_names read first
        self.measurements = measurements
        self.covariances = covariances

    @property
    def observable_names(self):
        return self._observable_names

    @observable_names.setter
    def observable_names(self, observable_names):
        assert isinstance(observable_names, (list,tuple))
        self._observable_names = tuple(observable_names)

    @property
    def measurements(self):
        return self._measurements

    @measurements.setter
    def measurements(self, measurements):
        assert (tuple(measurements.keys()) == self.observable_names)
        self._measurements = dict()
        for k in measurements.keys():
            self._measurements[k] = self._strip_data(measurements[k])

    @property
    def covariances(self):
        return self._covariances

    @covariances.setter
    def covariances(self, covariances):
        assert (tuple(covariances.keys()) == self.observable_names)
        self._covariances = dict()
        for k in covariances.keys():
            self._covariances[k] = self._strip_data(covariances[k])
    
    # notice the argument should be a dict of Observable objects
    def __call__(self, observables):
        assert (tuple(observables.keys()) == self.observable_names)
        likelicache = float(0)
        for name in self.observable_names:#{
            obs = observables[name]
            assert isinstance(obs,Observable)
            obs_mean = obs.ensemble_mean()
            data = deepcopy(self.measurements[name])
            diff = data - obs_mean
            cov = deepcopy(self.covariances[name])
            if cov is None: # no covariance matrix
                likelicache += -float(0.5)*float(np.vdot(diff,diff))
            else:
                (sign,logdet) = np.linalg.slogdet(cov*2.*np.pi)
                likelicache += -float(0.5)*float(np.vdot(diff,np.linalg.solve(cov,diff))+sign*logdet)
        #}
        return likelicache
