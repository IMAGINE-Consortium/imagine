"""
hammurabiX analytic CRE field
"""

import logging as log

from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy


@icy
class CREAna(GeneralField):

    def __init__(self, parameters=dict(), ensemble_size=1, random_seed=None):
        super(CREAna, self).__init__(parameters, ensemble_size, random_seed)
        self.name = 'cre_ana'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters
        :return: dict of XML locations
        """
        checklist = {'alpha': (['cre', 'analytic', 'alpha'], 'value'),
                     'beta': (['cre', 'analytic', 'beta'], 'value'),
                     'theta': (['cre', 'analytic', 'theta'], 'value'),
                     'r0': (['cre', 'analytic', 'r0'], 'value'),
                     'z0': (['cre', 'analytic', 'z0'], 'value'),
                     'E0': (['cre', 'analytic', 'E0'], 'value'),
                     'j0': (['cre', 'analytic', 'j0'], 'value')}
        return checklist

    @property
    def field_controllist(self):
        """
        record XML location of logical parameters
        :return: dict of XML locations
        """
        controllist = {'cue': (['cre'], {'cue': '1'}),
                       'type': (['cre'], {'type': 'analytic'})}
        return controllist
