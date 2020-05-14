from imagine import DummyField, GeneralFieldFactory, FlatPrior
from imagine.tools.icy_decorator import icy

@icy
class BrndES(DummyField):
    """
    This dummy field instructs the :py:class:`Hammurabi <imagine.simulators.hammurabi.Hammurabi>`
    simulator class to use the HammurabiX's builtin random magnetic field
    ES random GMF
    """
    field_name = 'breg_wmap'

    @property
    def field_checklist(self):
        """
        Hammurabi XML locations of physical parameters
        """
        checklist = {'rms': (['magneticfield', 'random', 'global', 'es', 'rms'], 'value'),
                     'k0': (['magneticfield', 'random', 'global', 'es', 'k0'], 'value'),
                     'a0': (['magneticfield', 'random', 'global', 'es', 'a0'], 'value'),
                     'k1': (['magneticfield', 'random', 'global', 'es', 'k1'], 'value'),
                     'a1': (['magneticfield', 'random', 'global', 'es', 'a1'], 'value'),
                     'rho': (['magneticfield', 'random', 'global', 'es', 'rho'], 'value'),
                     'r0': (['magneticfield', 'random', 'global', 'es', 'r0'], 'value'),
                     'z0': (['magneticfield', 'random', 'global', 'es', 'z0'], 'value'),
                     'random_seed': (['magneticfield', 'random'], 'seed'),
                    }
        return checklist

    @property
    def field_controllist(self):
        """
        Hammurabi XML locations of logical parameters
        """
        controllist = {'cue': (['magneticfield', 'random'], {'cue': '1'}),
                       'type': (['magneticfield', 'random'], {'type': 'global'}),
                       'method': (['magneticfield', 'random', 'global'], {'type': 'es'})}
        return controllist


@icy
class BrndESFactory(GeneralFieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BrndES`
    (see its docs for details).
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super().__init__(boxsize, resolution)
        self.field_class = BrndES
        self.default_parameters = {'rms': 2.,
                                   'k0': 10.0,
                                   'a0': 1.7,
                                   'k1': 0.1,
                                   'a1': 0.0,
                                   'rho': 0.5,
                                   'r0': 8.,
                                   'z0': 1. }
        self.priors = {'rms': FlatPrior([0, 4.]),
                       'k0': FlatPrior([0.1, 1.]),
                       'a0': FlatPrior([1., 3.]),
                       'k1': FlatPrior([0.01, 1.]),
                       'a1': FlatPrior([0., 3.]),
                       'rho': FlatPrior([0., 1.]),
                       'r0': FlatPrior([2., 10.]),
                       'z0': FlatPrior([0.1, 3.])}
        self.active_parameters = active_parameters
