# %% IMPORTS
import astropy.units as u
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import FlatPrior


# All declaration
__all__ = ['TEregUnif', 'TEregUnifFactory']


# %% CLASS DEFINITIONS
class TEregUnif(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's thermal electron density model YMW16
    """
    # Class attributes
    NAME = 'tereg_unif'
    FIELD_CHECKLIST = {
        'n0': (['thermalelectron', 'regular', 'unif', 'n0'],
               'value'),
        'r0': (['thermalelectron', 'regular', 'unif', 'r0'],
               'value'),
    }
    SIMULATOR_CONTROLLIST = {'cue': (['thermalelectron', 'regular'],
                                     {'cue': '1'}),
                             'type': (['thermalelectron', 'regular'],
                                      {'type': 'unif'})}


class TEregUnifFactory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`TEregYMW16`
    (see its docs for details).
    """
    # Class attributes
    FIELD_CLASS = TEregUnif
    DEFAULT_PARAMETERS = {'n0': 0.01*u.cm**3, 'r0': 3*u.pc}
    PRIORS = {'n0': FlatPrior(0, 1, unit=u.cm**3), 'r0': FlatPrior(0, 6, unit=u.pc)}
