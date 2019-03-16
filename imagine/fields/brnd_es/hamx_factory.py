"""
hammurabiX ES random GMF factory
"""

import logging as log

from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.brnd_es.hamx_field import BrndES
from imagine.tools.icy_decorator import icy


@icy
class BrndESFactory(GeneralFieldFactory):

    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(BrndESFactory, self).__init__(boxsize, resolution)
        self.field_type = 'vector'
        self.name = 'brnd_es'
        self.field_class = BrndES
        self.default_parameters = {'rms': 2., 'k0': 0.1, 'a0': 1.7, 'rho': 0.5, 'r0': 8., 'z0': 1.}
        self.parameter_ranges = {'rms': [0, 4.],
                                 'k0': [0.1, 1.],
                                 'a0': [1., 3.],
                                 'rho': [0., 1.],
                                 'r0': [2., 10.],
                                 'z0': [0.1, 3.]}
        self.active_parameters = active_parameters
