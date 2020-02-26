import logging as log
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.test_field.test_field import TestField
from imagine.tools.icy_decorator import icy


@icy
class TestFieldFactory(GeneralFieldFactory):

    def __init__(self, boxsize=None, resolution=None, active_parameters=tuple()):
        super(TestFieldFactory, self).__init__(boxsize, resolution)
        self.field_type = 'scalar'
        self.name = 'test'
        self.field_class = TestField
        self.default_parameters = {'a': 6.0,
                                   'b': 0.0}  # by default, no random field
        self.parameter_ranges = {'a': self._interval(6., 2., 3),
                                 'b': self._positive_interval(2.0, 1.0, 2.5)}
        self.active_parameters = active_parameters
