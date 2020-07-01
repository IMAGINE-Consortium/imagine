import numpy as np
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.field import GeneralField
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.fields.test_field.test_field import TestField


class TestFields():
    def test_generalfield_init(self):
        field = GeneralField({}, 2)
        assert field.name == 'general'
        assert field.field_checklist == {}
        assert field.ensemble_size == 2
        assert field.parameters == {}

    def test_testfield_init(self):
        parameters = {'a': 3, 'b': 4.2}
        size = 128
        seed_list = np.random.randint(1, 300, 128)
        field = TestField(parameters, size, seed_list)
        assert field.name == 'test'
        assert (field.field_checklist ==
                {'a': (['key', 'chain'], 'attribute'),
                 'random_seed': (['key', 'chain'], 'attribute'),
                 'b': (['key', 'chain'], 'attribute')})
        assert field.ensemble_size == 128
        assert (field.parameters ==  # no random_seed attached
                {'a': 3,
                 'b': 4.2})
        assert (field.report_parameters() ==  # return for 0th realization
                {'a': 3,
                 'b': 4.2,
                 'random_seed': seed_list[0]})
        assert (field.report_parameters(56) ==  # return for 57th realization
                {'a': 3,
                 'b': 4.2,
                 'random_seed': seed_list[56]})

    def test_empty_generalfactory_init(self):
        factory = GeneralFieldFactory()
        assert factory.boxsize is None
        assert factory.resolution is None

    def test_generalfactory_init(self):
        boxsize = (10, 20, 30)
        resolution = [5, 4, 6]
        factory = GeneralFieldFactory(boxsize, resolution)
        assert factory.field_type == 'scalar'
        assert factory.name == 'general'
        assert factory.field_class is GeneralField
        assert factory.boxsize == boxsize
        assert factory.resolution == (5, 4, 6)
        assert factory.active_parameters == ()
        assert factory.default_parameters == {}
        assert factory.parameter_ranges == {}
        assert factory.default_variables == {}

    def test_empty_testfactory_init(self):
        factory = TestFieldFactory()
        assert factory.boxsize is None
        assert factory.resolution is None

    def test_testfactory_init(self):
        boxsize = (10, 20, 30)
        resolution = [5, 4, 6]
        factory = TestFieldFactory(boxsize, resolution, ('a',))
        assert factory.name == 'test'
        assert factory.field_class is TestField
        assert factory.active_parameters == ('a',)
        assert (factory.default_parameters ==
                {'a': 6.0,
                 'b': 0.0})
        assert (factory.parameter_ranges ==
                {'a': (0, 12),
                 'b': (0, 4.5)})
        assert (factory.default_variables ==
                {'a': 0.5,
                 'b': 0})

        field = factory.generate({'a': 0.1}, 4, [23]*4)  # seed list by hand

        assert (field.report_parameters() ==
                {'a': 1.2,
                 'b': 0,
                 'random_seed': 23})
        assert field.ensemble_size == 4

    def test_testfactory_light_init(self):
        factory = TestFieldFactory(active_parameters=('a',))
        assert factory.name == 'test'
        assert factory.field_class is TestField
        assert factory.active_parameters == ('a',)
        assert (factory.default_parameters ==
                {'a': 6.0,
                 'b': 0.0})
        assert (factory.parameter_ranges ==
                {'a': (0, 12),
                 'b': (0, 4.5)})
        assert (factory.default_variables ==
                {'a': 0.5,
                 'b': 0})

        field = factory.generate({'a': 0.1}, 4)  # no seeds given, fully time-thread dependent random

        assert (field.parameters ==
                {'a': 1.2,
                 'b': 0})

        assert (field.report_parameters() ==
                {'a': 1.2,
                 'b': 0,
                 'random_seed': 0})
        assert field.ensemble_size == 4
