# %% IMPORTS
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import FlatPrior

# All declaration
__all__ = ['BregLSA', 'BregLSAFactory']


# %% CLASS DEFINITIONS
class BregLSA(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin regular magnetic field
    WMAP-3yr LSA.
    """

    # Class attributes
    NAME = 'breg_lsa'
    FIELD_CHECKLIST = {'b0': (['magneticfield', 'regular', 'lsa', 'b0'],
                              'value'),
                       'psi0': (['magneticfield', 'regular', 'lsa', 'psi0'],
                                'value'),
                       'psi1': (['magneticfield', 'regular', 'lsa', 'psi1'],
                                'value'),
                       'chi0': (['magneticfield', 'regular', 'lsa', 'chi0'],
                                'value')}
    SIMULATOR_CONTROLLIST = {'cue': (['magneticfield', 'regular'],
                                     {'cue': '1'}),
                             'type': (['magneticfield', 'regular'],
                                      {'type': 'lsa'})}


class BregLSAFactory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BregLSA`
    (see its docs for details).
    """

    # Class attributes
    FIELD_CLASS = BregLSA
    DEFAULT_PARAMETERS = {'b0': 6.0,
                          'psi0': 27.0,
                          'psi1': 0.9,
                          'chi0': 25.0}
    PRIORS = {'b0': FlatPrior([0., 10.]),
              'psi0': FlatPrior([0., 50.]),
              'psi1': FlatPrior([0., 5.]),
              'chi0': FlatPrior([-25., 50.])}
