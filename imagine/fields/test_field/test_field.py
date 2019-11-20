"""
test field is designed purely for testing purpose
setup a scalar field f = a*x + gaussian_random_err(b)
with parameter set {a,b}, with b defining gaussian random err half-width
test simulator is able to read xml parameter file
and checklist is designed of form
{parameter name: (parameter xml path, parameter xml tag)}
"""

import logging as log
from imagine.fields.field import GeneralField
from imagine.tools.icy_decorator import icy


@icy
class TestField(GeneralField):

    def __init__(self, parameters=dict(), ensemble_size=1, ensemble_seeds=None):
        super(TestField, self).__init__(parameters, ensemble_size, ensemble_seeds)
        self.name = 'test'

    @property
    def field_checklist(self):
        checklist = {'a': (['key', 'chain'], 'attribute'),
                     'random_seed': (['key', 'chain'], 'attribute'),
                     'b': (['key', 'chain'], 'attribute')}
        return checklist
