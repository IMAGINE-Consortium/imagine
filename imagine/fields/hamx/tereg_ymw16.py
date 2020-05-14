from imagine import DummyField, GeneralFieldFactory, FlatPrior
from imagine.tools.icy_decorator import icy

@icy
class TEregYMW16(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's thermal electron density model YMW16
    """
    name = 'tereg_ymw16'

    @property
    def field_checklist(self):
        """
        Hammurabi XML locations of physical parameters
        """
        checklist = dict()
        return checklist

    @property
    def simulator_controllist(self):
        """
        Hammurabi XML locations of logical parameters
        """
        controllist = {'cue': (['thermalelectron', 'regular'], {'cue': '1'}),
                       'type': (['thermalelectron', 'regular'], {'type': 'ymw16'})}
        return controllist


@icy
class TEregYMW16Factory(GeneralFieldFactory):
    """
    Field factory that produces the dummy field :py:class:`TEregYMW16`
    (see its docs for details).
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(TEregYMW16Factory, self).__init__(boxsize, resolution)
        self.field_class = TEregYMW16
        self.default_parameters = {}
        self.priors = {}
        self.active_parameters = active_parameters
