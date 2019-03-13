import numpy as np
import logging as log

import pymultinest

from imagine.likelihoods.likelihood import Likelihood
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.priors.prior import Prior
from imagine.simulators.simulator import Simulator
from imagine.tools.carrier_mapper import unity_mapper
from imagine.tools.timer import Timer
from imagine.tools.icy_decorator import icy


@icy
class MultinestPipeline(object):

    def __init__(self, simulator, factory_list, likelihood, prior, ensemble_size=1):
        """
        initialize Bayesian analysis pipeline with pyMultinest
        :param simulator: Simulator object
        :param factory_list: list/tuple of factory objects
        :param likelihood: Likelihood object
        :param prior: Prior object
        :param ensemble_size: number of observable realizations to be generated in simulator

        hidden controllers:

        dynesty_parameter_dict
            -- extra parameters for controlling Dynesty
            i.e., 'nlive', 'bound', 'sample'
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
        self.active_parameters = tuple()
        self.active_ranges = dict()
        self.factory_list = factory_list
        self.simulator = simulator
        self.likelihood = likelihood
        self.prior = prior
        self.ensemble_size = ensemble_size
        self.sampling_controllers = dict()
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
        assert isinstance(parameter_list, (list, tuple))
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
        """
        extract active_parameters and their ranges from each factory
        notice that once done
        the parameter/variable ordering is fixed wrt factory ordering
        which is useful in recovering variable logic value for each factory
        :param factory_list:
        :return:
        """
        assert isinstance(factory_list, (list, tuple))
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
        log.debug('set ensemble size to %i' % int(ensemble_size))

    @property
    def sampling_controllers(self):
        return self._sampling_controllers

    @sampling_controllers.setter
    def sampling_controllers(self, pp_dict):
        try:
            self._sampling_controllers.update(pp_dict)
            log.debug('update pymultinest parameter %s' % str(pp_dict))
        except AttributeError:
            self._sampling_controllers = pp_dict
            log.debug('set pymultinest parameter %s' % str(pp_dict))

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
        """

        :return: pyMultinest sampling results
        """
        # run pymultinest
        results = pymultinest.solve(LogLikelihood=self._core_likelihood,
                                    Prior=self.prior,
                                    n_dims=len(self._active_parameters),
                                    **self._sampling_controllers)
        return results

    def _core_likelihood(self, cube):
        """
        log-likelihood calculator
        :param cube: list of variable values
        :return: log-likelihood value
        """
        #t = Timer()
        log.debug('sampler at %s' % str(cube))
        # security boundary check
        if np.any(cube > 1.) or np.any(cube < 0.):
            log.debug('cube %s requested. returned most negative possible number' % str(instant_cube))
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
            for i, av in enumerate(factory.active_parameters):
                variable_dict[av] = factory_cube[i]
            field_list += (factory.generate(variables=variable_dict,
                                            ensemble_size=self.ensemble_size,
                                            random_seed=self._random_seed),)
            log.debug('create '+factory.name+' field')
            head_idx = tail_idx
        assert(head_idx == tail_idx)
        assert(head_idx == len(self._active_parameters))
        # create observables from fresh fields
        #t.tick('simulator')
        observables = self._simulator(field_list)
        #t.tock('simulator')
        # apply mask
        #t.tick('mask')
        observables.apply_mask(self.likelihood.mask_dict)
        #t.tock('mask')
        log.debug('create observables')
        # add up individual log-likelihood terms
        #t.tick('likeli')
        current_likelihood = self.likelihood(observables)
        #t.tock('likeli')
        log.debug('calc instant likelihood')
        #print('timing results: \n %s' % str(t.record))
        # check likelihood value until negative (or no larger than given threshold)
        if self._check_threshold and current_likelihood > self._likelihood_threshold:
            raise ValueError('log-likelihood beyond threashould')
        return current_likelihood * self.likelihood_rescaler
