import logging as log
from imagine.fields.field import GeneralField
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.tools.icy_decorator import icy

@icy
class TEregYMW16(GeneralField):
    """
    hammurabiX regular TE field, YMW16 model
    hammurabi has default YMW16 parameter setting
    """
    field_type = 'dummy'
    name = 'tereg_ymw16'

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


@icy
class TEregYMW16Factory(GeneralFieldFactory):
    """
    hammurabiX regular TE factory, YMW16 model
    with default YMW16 parameter setting
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(TEregYMW16Factory, self).__init__(boxsize, resolution)
        self.field_class = TEregYMW16
        self.default_parameters = dict()
        self.parameter_ranges = dict()
        self.active_parameters = active_parameters
