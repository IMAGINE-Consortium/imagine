'''
field class register the full default parameter value
which is passed down from field factory
and prepare to hand in to simulators the default checklist is a dict

members:
.simulator_checklist
    -- dict, with parameter name as entry, parameter xml path as content
    defines the parameters to be checked out by simulator
    should be fixed upon class designing
    
'''

import logging as log

class GeneralField(object):

    '''
    parameters
        -- dict of full parameter set {name: value}
    ensemble_size
        -- number of realisations in field ensemble
        useful only when random field is active
    random_seed
        -- random seed for generating random field realisations (likely in simulators)
        useful only when random field is active
    '''
    def __init__(self, parameters=dict(), ensemble_size=1, random_seed=None):
        self.parameters = parameters
        self.ensemble_size = ensemble_size
        # if checklist has 'random_seed' entry
        if 'random_seed' in self.simulator_checklist.keys():
            self.parameters.update({'random_seed':round(random_seed)})
            log.debug('update field random seed %s' % str(random_seed))
    
    @property
    def simulator_checklist(self):
        return dict()

    @property
    def ensemble_size(self):
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        assert (ensemble_size>0)
        self._ensemble_size = round(ensemble_size)
        log.debug('set field ensemble size %s' % str(ensemble_size))

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        for k in parameters:
            assert (k in self.simulator_checklist.keys())
        try:
            self._parameters
            self._parameters.update(parameters)
            log.debug('update full parameters %s' % str(parameters))
        except AttributeError:
            self._parameters = parameters
            log.debug('set full parameters %s' % str(parameters))
