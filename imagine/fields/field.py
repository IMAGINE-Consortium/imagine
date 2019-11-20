"""
this generalfield class register the full default parameter value
which is passed down from field factory
and prepare to hand in to simulators the default checklist is a dict.
"""

import logging as log
from imagine.tools.icy_decorator import icy


@icy
class GeneralField(object):

    def __init__(self, parameters=dict(), ensemble_size=1, ensemble_seeds=None):
        """

        parameters
        ----------
        
        parameters
            dict of full parameter set {name: value}
        
        ensemble_size
            number of realisations in field ensemble
        
        random_seed
            random seed for generating random field realisations
        """
        log.debug('@ field::__init__')
        self.name = 'general'
        self.parameters = parameters
        self.ensemble_size = ensemble_size
        self.ensemble_seeds = ensemble_seeds

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str)
        self._name = name
    
    @property
    def field_checklist(self):
        return dict()

    @property
    def ensemble_size(self):
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        assert (ensemble_size > 0)
        self._ensemble_size = round(ensemble_size)

    @property
    def ensemble_seeds(self):
        return self._ensemble_seeds

    @ensemble_seeds.setter
    def ensemble_seeds(self, ensemble_seeds):
        if ensemble_seeds is None:  # in case no seeds given, all 0
            self._ensemble_seeds = [int(0)]*self._ensemble_size
        else:
            assert (len(ensemble_seeds) == self._ensemble_size)
            self._ensemble_seeds = ensemble_seeds

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        for k in parameters:
            assert (k in self.field_checklist.keys())
        try:
            self._parameters.update(parameters)
            log.debug('update full-set parameters %s' % str(parameters))
        except AttributeError:
            self._parameters = parameters
            log.debug('set full-set parameters %s' % str(parameters))

    def report_parameters(self, realization_id=int(0)):
        """
        return parameters with random seed associated to realization id
        """
        log.debug('@ field::report_parameters')
        # if checklist has 'random_seed' entry
        if 'random_seed' in self.field_checklist.keys():
            self._parameters.update({'random_seed': self._ensemble_seeds[realization_id]})
        return self._parameters
