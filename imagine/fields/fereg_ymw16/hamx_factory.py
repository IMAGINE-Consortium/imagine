"""
hammurabiX regular FE factory, YMW16 model
hammurabiX has default YMW16 parameter setting
"""
import logging as log

from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.fereg_ymw16.hamx_field import FEregYMW16
from imagine.tools.icy_decorator import icy


@icy
class FEregYMW16Factory(GeneralFieldFactory):

    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(FEregYMW16Factory, self).__init__(boxsize, resolution)
        self.field_type = 'scalar'
        self.name = 'fereg_ymw16'
        self.field_class = FEregYMW16
        self.default_parameters = dict()
        self.parameter_ranges = dict()
        self.active_parameters = active_parameters
