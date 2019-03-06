"""
hammurabiX WMAP-3yr GMF
"""

import logging as log

from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy


@icy
class BregWMAP(GeneralField):

    def __init__(self, parameters=dict(), ensemble_size=1, random_seed=None):
        super(BregWMAP, self).__init__(parameters, ensemble_size, random_seed)
        self.name = 'breg_wmap'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters
        :return: dict of XML locations
        """
        checklist = {'b0': (['magneticfield', 'regular', 'wmap', 'b0'], 'value'),
                     'psi0': (['magneticfield', 'regular', 'wmap', 'psi0'], 'value'),
                     'psi1': (['magneticfield', 'regular', 'wmap', 'psi1'], 'value'),
                     'chi0': (['magneticfield', 'regular', 'wmap', 'chi0'], 'value')}
        return checklist

    @property
    def field_controllist(self):
        """
        record XML location of logical parameters
        :return: dict of XML locations
        """
        controllist = {'cue': (['magneticfield', 'regular'], {'cue': '1'}),
                       'type': (['magneticfield', 'regular'], {'type': 'wmap'})}
        return controllist
