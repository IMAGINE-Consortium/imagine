"""
hammurabiX ES random GMF
"""

import logging as log

from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy


@icy
class BrndES(GeneralField):

    def __init__(self, parameters=dict(), ensemble_size=1, random_seed=None):
        super(BrndES, self).__init__(parameters, ensemble_size, random_seed)
        self.name = 'breg_wmap'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters
        :return: dict of XML locations
        """
        checklist = {'rms': (['magneticfield', 'random', 'global', 'es', 'rms'], 'value'),
                     'k0': (['magneticfield', 'random', 'global', 'es', 'k0'], 'value'),
                     'a0': (['magneticfield', 'random', 'global', 'es', 'a0'], 'value'),
                     'rho': (['magneticfield', 'random', 'global', 'es', 'rho'], 'value'),
                     'r0': (['magneticfield', 'random', 'global', 'es', 'r0'], 'value'),
                     'z0': (['magneticfield', 'random', 'global', 'es', 'z0'], 'value'),
                     'random_seed': (['magneticfield', 'random'], 'seed')}
        return checklist

    @property
    def field_controllist(self):
        """
        record XML location of logical parameters
        :return: dict of XML locations
        """
        controllist = {'cue': (['magneticfield', 'random'], {'cue': '1'}),
                       'type': (['magneticfield', 'random'], {'type': 'global'}),
                       'method': (['magneticfield', 'random', 'global'], {'type': 'es'})}
        return controllist
