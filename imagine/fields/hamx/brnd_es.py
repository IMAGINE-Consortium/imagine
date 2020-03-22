"""
hammurabiX ES random GMF
"""
import logging as log
from imagine.fields.field import GeneralField
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.tools.icy_decorator import icy


@icy
class BrndES(GeneralField):
    """
    hammurabiX ES random GMF
    """
    field_name = 'breg_wmap'
    field_type = 'dummy'

    @property
    def field_checklist(self):
        """
        record XML location of physical parameters

        return
        ------
        dict of XML locations
        """
        checklist = {'rms': (['magneticfield', 'random', 'global', 'es', 'rms'], 'value'),
                     'k0': (['magneticfield', 'random', 'global', 'es', 'k0'], 'value'),
                     'a0': (['magneticfield', 'random', 'global', 'es', 'a0'], 'value'),
                     'k1': (['magneticfield', 'random', 'global', 'es', 'k1'], 'value'),
                     'a1': (['magneticfield', 'random', 'global', 'es', 'a1'], 'value'),
                     'rho': (['magneticfield', 'random', 'global', 'es', 'rho'], 'value'),
                     'r0': (['magneticfield', 'random', 'global', 'es', 'r0'], 'value'),
                     'z0': (['magneticfield', 'random', 'global', 'es', 'z0'], 'value'),
                     'random_seed': (['magneticfield', 'random'], 'seed')}
        return checklist

    @property
    def field_controllist(self):
        """
        Dictionary of record XML locations of logical parameters
        """
        controllist = {'cue': (['magneticfield', 'random'], {'cue': '1'}),
                       'type': (['magneticfield', 'random'], {'type': 'global'}),
                       'method': (['magneticfield', 'random', 'global'], {'type': 'es'})}
        return controllist


@icy
class BrndESFactory(GeneralFieldFactory):
    """
    hammurabiX ES random GMF factory
    """
    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(BrndESFactory, self).__init__(boxsize, resolution)
        self.field_class = BrndES
        self.default_parameters = {'rms': 2., 'k0': 10.0, 'a0': 1.7, 'k1': 0.1,
                                   'a1': 0.0, 'rho': 0.5, 'r0': 8., 'z0': 1.}
        self.parameter_ranges = {'rms': [0, 4.],
                                 'k0': [0.1, 1.],
                                 'a0': [1., 3.],
                                 'k1': [0.01, 1.],
                                 'a1': [0., 3.],
                                 'rho': [0., 1.],
                                 'r0': [2., 10.],
                                 'z0': [0.1, 3.]}
        self.active_parameters = active_parameters
