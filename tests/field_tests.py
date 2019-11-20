import unittest
import numpy as np
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.field import GeneralField
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.fields.test_field.test_field import TestField


class TestFields(unittest.TestCase):
    
    def test_generalfield_init(self):
        field = GeneralField({}, 2.2)
        self.assertEqual(field.name, 'general')
        self.assertEqual(field.field_checklist, {})
        self.assertEqual(field.ensemble_size, round(2))
        self.assertEqual(field.parameters, {})
    
    def test_testfield_init(self):
        parameters = {'a': float(3), 'b': float(4.2)}
        size = 128
        seed_list = np.random.randint(1, 300, 128)
        field = TestField(parameters, size, seed_list)
        self.assertEqual(field.name, 'test')
        self.assertEqual(field.field_checklist,
                         {'a': (['key', 'chain'], 'attribute'),
                          'random_seed': (['key', 'chain'], 'attribute'),
                          'b': (['key', 'chain'], 'attribute')})
        self.assertEqual(field.ensemble_size, int(128))
        self.assertEqual(field.parameters,  # no random_seed attached
                         {'a': float(3),
                          'b': float(4.2)})
        self.assertEqual(field.report_parameters(),  # return for 0th realization
                         {'a': float(3),
                          'b': float(4.2),
                          'random_seed': seed_list[0]})
        self.assertEqual(field.report_parameters(56),  # return for 57th realization
                         {'a': float(3),
                          'b': float(4.2),
                          'random_seed': seed_list[56]})

    def test_empty_generalfactory_init(self):
        factory = GeneralFieldFactory()
        self.assertEqual(factory.boxsize, None)
        self.assertEqual(factory.resolution, None)
    
    def test_generalfactory_init(self):
        boxsize = (float(10), float(20), float(30))
        resolution = [round(5), round(4), round(6)]
        factory = GeneralFieldFactory(boxsize, resolution)
        self.assertEqual(factory.field_type, 'scalar')
        self.assertEqual(factory.name, 'general')
        self.assertEqual(factory.field_class, GeneralField)
        self.assertEqual(factory.boxsize, boxsize)
        self.assertEqual(factory.resolution, (round(5), round(4), round(6)))
        self.assertEqual(factory.active_parameters, ())
        self.assertEqual(factory.default_parameters, {})
        self.assertEqual(factory.parameter_ranges, {})
        self.assertEqual(factory.default_variables, {})

    def test_empty_testfactory_init(self):
        factory = TestFieldFactory()
        self.assertEqual(factory.boxsize, None)
        self.assertEqual(factory.resolution, None)
    
    def test_testfactory_init(self):
        boxsize = (float(10), float(20), float(30))
        resolution = [round(5), round(4), round(6)]
        factory = TestFieldFactory(boxsize, resolution, tuple('a'))
        self.assertEqual(factory.name, 'test')
        self.assertEqual(factory.field_class, TestField)
        self.assertEqual(factory.active_parameters, tuple('a'))
        self.assertEqual(factory.default_parameters, {'a': 6.0,
                                                      'b': 0.0})
        self.assertEqual(factory.parameter_ranges, {'a': (float(0), float(12)),
                                                    'b': (float(0), float(4.5))})
        self.assertEqual(factory.default_variables, {'a': float(6)/float(12),
                                                     'b': float(0)/float(4.5)})
        
        field = factory.generate({'a': 0.1}, 4, [23]*4)  # seed list by hand
                          
        self.assertEqual(field.report_parameters(), {'a': float(0.1)*float(12),
                                                     'b': float(0.0),
                                                     'random_seed': round(23)})
        self.assertEqual(field.ensemble_size, round(4))

    def test_testfactory_light_init(self):
        factory = TestFieldFactory(active_parameters=tuple('a'))
        self.assertEqual(factory.name, 'test')
        self.assertEqual(factory.field_class, TestField)
        self.assertEqual(factory.active_parameters, tuple('a'))
        self.assertEqual(factory.default_parameters, {'a': 6.0,
                                                      'b': 0.0})
        self.assertEqual(factory.parameter_ranges, {'a': (float(0), float(12)),
                                                    'b': (float(0), float(4.5))})
        self.assertEqual(factory.default_variables, {'a': float(6)/float(12),
                                                     'b': float(0)/float(4.5)})
        
        field = factory.generate({'a': 0.1}, 4)  # no seeds given, fully time-thread dependent random

        self.assertEqual(field.parameters, {'a': float(0.1)*float(12),
                                            'b': float(0.0)})
                          
        self.assertEqual(field.report_parameters(), {'a': float(0.1)*float(12),
                                                     'b': float(0.0),
                                                     'random_seed': int(0)})
        self.assertEqual(field.ensemble_size, round(4))


if __name__ == '__main__':
    unittest.main()
