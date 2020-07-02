# %% IMPORTS
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory

# All declaration
__all__ = ['TEregYMW16', 'TEregYMW16Factory']


# %% CLASS DEFINITIONS
class TEregYMW16(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's thermal electron density model YMW16
    """

    # Class attributes
    NAME = 'tereg_ymw16'

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


class TEregYMW16Factory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`TEregYMW16`
    (see its docs for details).
    """

    # Class attributes
    FIELD_CLASS = TEregYMW16
    DEFAULT_PARAMETERS = {}
    PRIORS = {}
