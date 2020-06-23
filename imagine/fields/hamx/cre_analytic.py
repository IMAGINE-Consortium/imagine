from imagine import DummyField, GeneralFieldFactory, FlatPrior
from imagine.tools.icy_decorator import icy

@icy
class CREAna(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin analytic cosmic ray electron
    distribution
    """
    field_name = 'cre_ana'

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


@icy
class CREAnaFactory(GeneralFieldFactory):
    """
    Field factory that produces the dummy field :py:class:`CREAna`
    (see its docs for details).
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super().__init__(boxsize, resolution)
        self.field_class = CREAna
        self.default_parameters = {'alpha': 3.0,
                                   'beta': 0.0,
                                   'theta': 0.0,
                                   'r0': 5.0,
                                   'z0': 1.0,
                                   'E0': 20.6,
                                   'j0': 0.0217}
        self.priors = {'alpha': FlatPrior([2., 4.]),
                       'beta': FlatPrior([-1., 1.]),
                       'theta': FlatPrior([-1., 1.]),
                       'r0': FlatPrior([0.1, 10.]),
                       'z0': FlatPrior([0.1, 3.]),
                       'E0': FlatPrior([10., 30.]),
                       'j0': FlatPrior([0., 0.1])}
        self.active_parameters = active_parameters
