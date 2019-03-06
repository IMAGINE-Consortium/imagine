"""
hammurabiX analytical CRE factory
"""
import logging as log

from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.cre_analytic.hamx_field import CREAna
from imagine.tools.icy_decorator import icy


@icy
class CREAnaFactory(GeneralFieldFactory):

    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(CREAnaFactory, self).__init__(boxsize, resolution)
        self.field_type = 'scalar'
        self.name = 'cre_ana'
        self.field_class = CREAna
        self.default_parameters = {'alpha': 3.0,
                                   'beta': 0.0,
                                   'theta': 0.0,
                                   'r0': 5.0,
                                   'z0': 1.0,
                                   'E0': 20.6,
                                   'j0': 0.0217}
        self.parameter_ranges = {'alpha': [2., 4.],
                                 'beta': [-1., 1.],
                                 'theta': [-1., 1.],
                                 'r0': [0.1, 10.],
                                 'z0': [0.1, 3.],
                                 'E0': [10., 30.],
                                 'j0': [0., 0.1]}
        self.active_parameters = active_parameters
