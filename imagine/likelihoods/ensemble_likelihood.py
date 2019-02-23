'''
ensemble likelihood, described in IMAGINE techincal report
in principle
it adds covariance matrices from both observation and simulation
'''
import numpy as np
from copy import deepcopy

from imagine.observables.observable import Observable
from imagine.likelihoods.likelihood import Likelihood

class EnsembleLikelihood(Likelihood):
    
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
            # estimate simulational ensemble_mean, covariance
            (obs_mean,obs_cov) = self._oas(obs)
            # get observational info
            data = deepcopy(self.measurements[name])
            assert (len(data) == len(obs_mean))
            data_cov = deepcopy(self.covariance[name])
            if data_cov is not None:
                obs_cov = obs_cov + data_cov
            diff = data - obs_mean
            diff = np.nan_to_num(diff)
            # calc loglikeli
            (sign,logdet) = np.linalg.slogdet((obs_cov)*2.*np.pi)
            likelicache += float(-0.5)*float(np.vdot(diff,np.linalg.solve(obs_cov,diff))+sign*logdet)
        #}
        return likelicache

    # OAS estimator, observable comes with (ensemble_number,data_size) matrix
    # take Observable object as argument
    def _oas(observable):
        (n,p) = observable.shape
        assert (p>0)
        mean = observable.ensemble_mean().val.get_full_data()
        u = observable.val.get_full_data() - mean # should broadcast to all rows
        S = np.dot(np.transpose(u),u)/n # emprical covariance S
        TrS = np.trace(S) # Tr(S), equivalent to np.vdot(u,u)/n
        TrS2 = np.trace(np.dot(S,S)) # Tr(S^2), equivalent to (np.einsum(u,[0,1],u,[2,1])**2).sum() / (n**2)
        # calc rho
        numerator = (1-2./p)*TrS2 + TrS**2
        denominator = (n+1-2./p)*(TrS2-(TrS**2)/p)
        if denominator==0:
            rho = 1
        else:
            rho = np.min([1.,float(numerator/denominator)])	
        return (mean,(1.-rho)*S + np.eye(p)*rho*TrS/p)
