from imagine import DummyField, GeneralFieldFactory, FlatPrior
from imagine.tools.icy_decorator import icy

@icy
class BregLSA(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin regular magnetic field
    WMAP-3yr LSA.
    """
    field_name = 'breg_lsa'

    @property
    def field_checklist(self):
        """
        Hammurabi XML locations of physical parameters
        """
        checklist = {'b0': (['magneticfield', 'regular', 'lsa', 'b0'], 'value'),
                     'psi0': (['magneticfield', 'regular', 'lsa', 'psi0'], 'value'),
                     'psi1': (['magneticfield', 'regular', 'lsa', 'psi1'], 'value'),
                     'chi0': (['magneticfield', 'regular', 'lsa', 'chi0'], 'value')}
        return checklist

    @property
    def field_controllist(self):
        """
        Hammurabi XML locations of logical parameters
        """
        controllist = {'cue': (['magneticfield', 'regular'], {'cue': '1'}),
                       'type': (['magneticfield', 'regular'], {'type': 'lsa'})}
        return controllist


@icy
class BregLSAFactory(GeneralFieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BregLSA`
    (see its docs for details).
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super().__init__(boxsize, resolution)
        self.field_class = BregLSA
        self.default_parameters = {'b0': 6.0,
                                   'psi0': 27.0,
                                   'psi1': 0.9,
                                   'chi0': 25.0}
        self.priors = {'b0':   FlatPrior([0., 10.]),
                       'psi0': FlatPrior([0., 50.]),
                       'psi1': FlatPrior([0., 5.]),
                       'chi0': FlatPrior([-25., 50.])},
        self.active_parameters = active_parameters
