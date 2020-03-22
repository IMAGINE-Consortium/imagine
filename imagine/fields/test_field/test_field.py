import logging as log
from imagine.fields import GeneralField
from imagine.fields import GeneralFieldFactory
from imagine.tools.icy_decorator import icy


@icy
class TestField(GeneralField):
    """
    test field is designed purely for testing purpose
    setup a scalar field f = a*x + gaussian_random_err(b)
    with parameter set {a,b}, with b defining gaussian random err half-width
    test simulator is able to read xml parameter file
    and checklist is designed of form
    {parameter name: (parameter xml path, parameter xml tag)}
    """
    field_name = 'test'
    field_type = 'dummy'

    @property
    def field_checklist(self):
        checklist = {'a': (['key', 'chain'], 'attribute'),
                     'random_seed': (['key', 'chain'], 'attribute'),
                     'b': (['key', 'chain'], 'attribute')}
        return checklist

# @icy
class TestFieldFactory(GeneralFieldFactory):
    def __init__(self, boxsize=None, resolution=None,
                 active_parameters=tuple()):
        super(TestFieldFactory, self).__init__(boxsize, resolution)
        self.field_class = TestField
        self.default_parameters = {'a': 6.0,
                                   'b': 0.0}  # by default, no random field
        self.parameter_ranges = {'a': self._interval(6., 2., 3),
                                 'b': self._positive_interval(2.0,1.0,2.5)}
        self.active_parameters = active_parameters
