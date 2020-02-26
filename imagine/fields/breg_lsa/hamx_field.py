"""
hammurabiX WMAP-3yr LSA GMF
"""

import logging as log
from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy


@icy
class BregLSA(GeneralField):

    def __init__(self, parameters=dict(), ensemble_size=1, ensemble_seeds=None):
        super(BregLSA, self).__init__(parameters, ensemble_size, ensemble_seeds)
        self.name = 'breg_lsa'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters
        
        return
        ------
        dict of XML locations
        """
        checklist = {'b0': (['magneticfield', 'regular', 'lsa', 'b0'], 'value'),
                     'psi0': (['magneticfield', 'regular', 'lsa', 'psi0'], 'value'),
                     'psi1': (['magneticfield', 'regular', 'lsa', 'psi1'], 'value'),
                     'chi0': (['magneticfield', 'regular', 'lsa', 'chi0'], 'value')}
        return checklist

    @property
    def field_controllist(self):
        """
        record XML location of logical parameters
        
        return
        ------
        dict of XML locations
        """
        controllist = {'cue': (['magneticfield', 'regular'], {'cue': '1'}),
                       'type': (['magneticfield', 'regular'], {'type': 'lsa'})}
        return controllist
