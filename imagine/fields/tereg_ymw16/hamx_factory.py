"""
hammurabiX regular TE factory, YMW16 model
hammurabiX has default YMW16 parameter setting
"""
import logging as log

from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.tereg_ymw16.hamx_field import TEregYMW16
from imagine.tools.icy_decorator import icy


@icy
class TEregYMW16Factory(GeneralFieldFactory):
    """
    hammurabiX regular TE factory, YMW16 model
    with default YMW16 parameter setting
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(TEregYMW16Factory, self).__init__(boxsize, resolution)
        self.field_type = 'scalar'
        self.name = 'tereg_ymw16'
        self.field_class = TEregYMW16
        self.default_parameters = dict()
        self.parameter_ranges = dict()
        self.active_parameters = active_parameters
