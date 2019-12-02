"""
hammurabiX regular TE field, YMW16 model
hammurabi has default YMW16 parameter setting
"""

import logging as log
from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy


@icy
class TEregYMW16(GeneralField):

    def __init__(self, parameters=dict(), ensemble_size=1, ensemble_seeds=None):
        super(TEregYMW16, self).__init__(parameters, ensemble_size, ensemble_seeds)
        self.name = 'tereg_ymw16'

    @property
    def field_checklist(self):
        """
        Dictionary of XML location of physical parameters
        """
        checklist = dict()
        return checklist

    @property
    def field_controllist(self):
        """
        Dictionary of XML location of logical parameters
        """
        controllist = {'cue': (['thermalelectron', 'regular'], {'cue': '1'}),
                       'type': (['thermalelectron', 'regular'], {'type': 'ymw16'})}
        return controllist
