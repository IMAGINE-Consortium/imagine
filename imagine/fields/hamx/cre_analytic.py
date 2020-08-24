# %% IMPORTS
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import FlatPrior

# All declaration
__all__ = ['CREAna', 'CREAnaFactory']


# %% CLASS DEFINITIONS
class CREAna(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin analytic cosmic ray electron
    distribution
    """
    NAME = 'cre_ana'

    @property
    def field_checklist(self):
        """
        Hammurabi XML locations of physical parameters
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
    def simulator_controllist(self):
        """
        Hammurabi XML locations of logical parameters
        """
        controllist = {'cue': (['cre'], {'cue': '1'}),
                       'type': (['cre'], {'type': 'analytic'})}
        return controllist


class CREAnaFactory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`CREAna`
    (see its docs for details).
    """

    # Class attributes
    FIELD_CLASS = CREAna
    DEFAULT_PARAMETERS = {'alpha': 3,
                          'beta': 0,
                          'theta': 0,
                          'r0': 5,
                          'z0': 1,
                          'E0': 20.6,
                          'j0': 0.0217}
    PRIORS = {'alpha': FlatPrior(xmin=2, xmax=4),
              'beta': FlatPrior(xmin=-1, xmax=1),
              'theta': FlatPrior(xmin=-1, xmax=1),
              'r0': FlatPrior(xmin=0.1, xmax=10),
              'z0': FlatPrior(xmin=0.1, xmax=3),
              'E0': FlatPrior(xmin=10, xmax=30),
              'j0': FlatPrior(xmin=0, xmax=0.1)}
