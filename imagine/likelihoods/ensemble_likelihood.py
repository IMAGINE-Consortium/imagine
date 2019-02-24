'''
ensemble likelihood, described in IMAGINE techincal report
in principle
it adds covariance matrices from both observation and simulation
'''
import numpy as np
from copy import deepcopy

from imagine.observables.observable import Observable
from imagine.observables.observable_dict import Measurements, Simulations, Covariances
from imagine.likelihoods.likelihood import Likelihood

class EnsembleLikelihood(Likelihood):
    
    '''
    measurement_dict
        -- Measurements object
    covariance_dict
        -- Covariances object
    observable_dict (in __call__)
        -- Simulations object
    '''
    def __init__(self, measurement_dict, covariance_dict=None):
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
    
    # notice the argument should be a dict of Observable objects
    def __call__(self, observable_dict):
        # check dict entries
        assert (observable_dict.keys() == self._measurement_dict.keys())
        likelicache = float(0)
        if self._covariance_dict is None:
            for name in self._measurement_dict.keys():
                (obs_mean,obs_cov) = self._oas(observable_dict[name])
                data = deepcopy(self._measurement_dict[name].to_global_data())
                diff = np.nan_to_num(data - obs_mean)
                (sign,logdet) = np.linalg.slogdet(obs_cov*2.*np.pi)
                likelicache += -float(0.5)*float(np.vdot(diff,np.linalg.solve(obs_cov,diff.T))+sign*logdet)
        else:
            for name in self._measurement_dict.keys():
                (obs_mean,obs_cov) = self._oas(observable_dict[name])
                data = deepcopy(self._measurement_dict[name].to_global_data())
                full_cov = deepcopy(self._covariance_dict[name].to_global_data()) + obs_cov
                diff = np.nan_to_num(data - obs_mean)
                (sign,logdet) = np.linalg.slogdet(full_cov*2.*np.pi)
                likelicache += -float(0.5)*float(np.vdot(diff,np.linalg.solve(full_cov,diff.T))+sign*logdet)
        return likelicache

    # OAS estimator, observable comes with (ensemble_number,data_size) matrix
    # take Observable object as input
    def _oas(self, observable):
        assert isinstance(observable, Observable)
        (n,p) = observable.shape
        assert (p>0 and n>1)
        mean = observable.ensemble_mean
        u = observable.to_global_data() - mean # should broadcast to all rows
        S = np.dot(u.T,u)/n # emprical covariance S
        assert (S.shape[0] == u.shape[1])
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
