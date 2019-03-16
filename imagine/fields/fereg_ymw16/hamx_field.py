"""
hammurabiX regular FE field, YMW16 model
hammurabi has default YMW16 parameter setting
"""

import logging as log

from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy


@icy
class FEregYMW16(GeneralField):

    def __init__(self, parameters=dict(), ensemble_size=1, ensemble_seeds=None):
        super(FEregYMW16, self).__init__(parameters, ensemble_size, ensemble_seeds)
        self.name = 'fereg_ymw16'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters
        :return: dict of XML locations
        """
        checklist = dict()
        return checklist

    @property
    def field_controllist(self):
        """
        record XML location of logical parameters
        :return: dict of XML locations
        """
        controllist = {'cue': (['freeelectron', 'regular'], {'cue': '1'}),
                       'type': (['freeelectron', 'regular'], {'type': 'ymw16'})}
        return controllist
