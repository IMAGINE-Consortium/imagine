import logging as log

from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.breg_wmap.hamx_field import BregWMAP
from imagine.tools.icy_decorator import icy


@icy
class BregWMAPFactory(GeneralFieldFactory):

    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(TestFieldFactory, self).__init__(boxsize, resolution)
        self.field_type = 'vector'
        self.name = 'wmap'
        self.field_class = BregWMAP
        self.default_parameters = {'b0': 6.0, 'psi0': 27.0, 'psi1': 1.0, 'chi0': 25.0}
        self.parameter_ranges = {'b0': self._interval(6., 2., 3),
                                 'psi0': self._interval(30.0, 20.0, 3),
                                 'psi1': self._interval(1.0, 5.0, 3),
                                 'chi0': self._interval(25.0, 20.0, 3)}
        self.active_parameters = active_parameters
