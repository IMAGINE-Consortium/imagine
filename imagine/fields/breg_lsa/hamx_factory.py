"""
hammurabiX WMAP-3yr LSA GMF factory
"""

import logging as log
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.breg_lsa.hamx_field import BregLSA
from imagine.tools.icy_decorator import icy


@icy
class BregLSAFactory(GeneralFieldFactory):

    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(BregLSAFactory, self).__init__(boxsize, resolution)
        self.field_type = 'vector'
        self.name = 'breg_lsa'
        self.field_class = BregLSA
        self.default_parameters = {'b0': 6.0, 'psi0': 27.0, 'psi1': 0.9, 'chi0': 25.0}
        self.parameter_ranges = {'b0': [0., 10.],
                                 'psi0': [0., 50.],
                                 'psi1': [0., 5.],
                                 'chi0': [-25., 50.]}
        self.active_parameters = active_parameters
