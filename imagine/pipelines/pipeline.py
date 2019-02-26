import os
import shutil
import json
import numpy as np
import logging as log

import pymultinest

from imagine.likelihoods.likelihood import Likelihood
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.priors.prior import Prior
from imagine.simulators.simulator import Simulator
from imagine.tools.carrier_mapper import unity_mapper

class Pipeline(object):

    """
    simulator
        -- as you would imagine it should be
    factory_list
        -- list/tuple of factory objects
    likelihood
        -- Likelihood object
    prior
        -- Prior object
    ensemble_size
        -- number of observable realizations to be generated
        in simulator

    hidden controllers:

    pymultinest_parameter_dict
        -- extra parameters for running pymultinest routine
    sample_callback
        -- not implemented yet
    likelihood_rescaler
        -- rescale log-likelihood value
    random_seed
        -- if 0 (default), use time-thread dependent random seed in simulator
        costomised value should be positive int
    likelihood_threshold
        -- by default, log-likelihood should be negative
    """
    def __init__(self, simulator, factory_list, likelihood, prior, ensemble_size=1):
        self.active_parameters = tuple()
        self.active_ranges = dict()
        self.factory_list = factory_list
        self.simulator = simulator
        self.likelihood = likelihood
        self.prior = prior
        self.ensemble_size = ensemble_size
        #
        # hidden controllers :)
        #
        # setting defaults for pymultinest
        self.pymultinest_parameter_dict = {'verbose': True,
                                           'n_iter_before_update': 100,
                                           'n_live_points': 400,
                                           'resume': False}
        self.sample_callback = False
        # rescaling total likelihood in _core_likelihood
        self.likelihood_rescaler = 1.
        # using fixed seed or time-thread dependent seed
        self.random_seed = 0
        # checking likelihood threshold
        self.check_threshold = False
        self.likelihood_threshold = 0.

    @property
    def active_parameters(self):
        return self._active_parameters

    @active_parameters.setter
    def active_parameters(self, parameter_list):
        assert isinstance(parameter_list, (list,tuple))
        self._active_parameters = tuple(parameter_list)

    @property
    def active_ranges(self):
        return self._active_ranges

    @active_ranges.setter
    def active_ranges(self, active_ranges):
        assert isinstance(active_ranges, dict)
        self._active_ranges = active_ranges

    @property
    def factory_list(self):
        return self._factory_list

    @factory_list.setter
    def factory_list(self, factory_list):
        assert isinstance(factory_list, (list,tuple))
        # extract active_parameters and their ranges from each factory
        # notice that once done
        # the parameter/variable ordering is fixed wrt factory ordering
        # which is useful in recovering variable logic value for each factory
        for factory in factory_list:
            assert isinstance(factory, GeneralFieldFactory)
            for ap_name in factory.active_parameters:
                assert isinstance(ap_name, str)
                self._active_parameters += (str(factory.name+'_'+ap_name),)
                self._active_ranges[str(factory.name+'_'+ap_name)] = factory.parameter_ranges[ap_name]
        self._factory_list = factory_list

    @property
    def simulator(self):
        return self._simulator

    @simulator.setter
    def simulator(self, simulator):
        assert isinstance(simulator, Simulator)
        self._simulator = simulator

    @property
    def likelihood(self):
        return self._likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        assert isinstance(likelihood, Likelihood)
        self._likelihood = likelihood
    
    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, prior):
        assert isinstance(prior, Prior)
        self._prior = prior

    @property
    def ensemble_size(self):
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        ensemble_size = int(ensemble_size)
        assert (ensemble_size > 0)
        self._ensemble_size = ensemble_size
        log.debug ('set ensemble size to %i' % int(ensemble_size))

    @property
    def pymultinest_parameter_dict(self):
        return self._pymultinest_parameter_dict

    @pymultinest_parameter_dict.setter
    def pymultinest_parameter_dict(self, pp_dict):
        try:
            self._pymultinest_parameter_dict.update(pp_dict)
            log.debug ('update pymultinest parameter %s' % str(pp_dict))
        except AttributeError:
            self._pymultinest_parameter_dict = pp_dict
            log.debug ('set pymultinest parameter %s' % str(pp_dict))

    @property
    def sample_callback(self):
        return self._sample_callback

    @sample_callback.setter
    def sample_callback(self, sample_callback):
        self._sample_callback = sample_callback

    @property
    def likelihood_rescaler(self):
        return self._likelihood_rescaler

    @likelihood_rescaler.setter
    def likelihood_rescaler(self, likelihood_rescaler):
        self._likelihood_rescaler = likelihood_rescaler

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        assert isinstance(random_seed, int)
        self._random_seed = random_seed

    @property
    def check_threshold(self):
        return self._check_threshold

    @check_threshold.setter
    def check_threshold(self, check_threshold):
        self._check_threshold = check_threshold

    @property
    def likelihood_threshold(self):
        return self._likelihood_threshold

    @likelihood_threshold.setter
    def likelihood_threshold(self, likelihood_threshold):
        self._likelihood_threshold = likelihood_threshold

    def __call__(self):
        # create dir for storing pymultinest output
        path = os.path.join(os.getcwd(),'chains')
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)
        assert (os.path.exists(path))
        json.dump(self._active_parameters,open('chains/imagine_params.json','w'))
        # run pymultinest
        result = pymultinest.solve(LogLikelihood=self._core_likelihood,
                                   Prior=self.prior,
                                   n_dims=len(self._active_parameters),
                                   outputfiles_basename='chains/imagine_',
                                   **self._pymultinest_parameter_dict)
        print ('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print ('variable values:')
        for name, col in zip(self._active_parameters, result['samples'].transpose()):
            print ('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            """
            # we think it's better to display raw variable values
                                               #unity_mapper(col.mean(),
                                               #             self._active_ranges[name][0],
                                               #             self._active_ranges[name][1]),
                                               #unity_mapper(col.std(),
                                               #             self._active_ranges[name][0],
                                               #             self._active_ranges[name][1])))
            """
        print ('detailed results dumped in %s' % str(path))

    """
    log-likelihood calculator
    """
    def _core_likelihood(self, cube):
        log.debug('sampler at %s' % str(cube))
        # security boundary check
        if np.any(cube > 1.) or np.any(cube < 0.):
            log.debug ('cube %s requested. returned most negative possible number' % str(instant_cube))
            return np.nan_to_num(-np.inf)
        # return active variables from pymultinest cube to factories
        # and then generate new field objects
        head_idx = int(0)
        tail_idx = int(0)
        field_list = tuple()
        # the ordering in factory list and variable list is vital
        for factory in self._factory_list:
            variable_dict = dict()
            tail_idx = head_idx + len(factory.active_parameters)
            factory_cube = cube[head_idx:tail_idx]
            for i,av in enumerate(factory.active_parameters):
                variable_dict[av] = factory_cube[i]
            field_list += (factory.generate(variables=variable_dict,
                                            ensemble_size=self.ensemble_size,
                                            random_seed=self._random_seed),)
            log.debug ('create '+factory.name+' field')
            head_idx += tail_idx
        assert(head_idx == tail_idx)
        assert(head_idx == len(self._active_parameters))
        # create observables from fresh fields
        observables = self._simulator(field_list)
        log.debug ('create observables')
        # add up individual log-likelihood terms
        current_likelihood = self.likelihood(observables)
        log.debug ('calc instant likelihood')
        # check likelihood value until negative (or no larger than given threshold)
        if self._check_threshold and current_likelihood > self._likelihood_threshold:
            raise ValueError('log-likelihood beyond threashould')
        return current_likelihood * self.likelihood_rescaler