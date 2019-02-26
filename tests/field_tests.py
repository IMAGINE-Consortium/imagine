"""
check functions as much as possible in base classes
"""
import unittest
from imagine.fields.field_factory import GeneralFieldFactory
from imagine.fields.field import GeneralField
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.fields.test_field.test_field import TestField

class FieldTests(unittest.TestCase):
    
    def test_generalfield_init(self):
        gf = GeneralField({},2.2)
        # test common base class init
        self.assertEqual (gf.name, 'general')
        self.assertEqual (gf.field_checklist, {})
        self.assertEqual (gf.ensemble_size, round(2))
        self.assertEqual (gf.parameters, {})
    
    def test_testfield_init(self):
        test_list = {'a':float(3),'b':float(4.2)}
        tf = TestField(test_list,128.3,34.8)
        # test testfield init
        self.assertEqual (tf.name, 'test')
        self.assertEqual (tf.field_checklist,
                          {'a': ('dummypath','dummy'),
                           'random_seed': ('dummypath','dummy'),
                           'b': ('dummypath','dummy')})
        self.assertEqual (tf.ensemble_size, round(128.3))
        self.assertEqual (tf.parameters,
                          {'a':float(3),
                           'b':float(4.2),
                           'random_seed':round(34.8)})

    def test_empty_generalfactory_init(self):
        gff_empty = GeneralFieldFactory()
        # test empty factory
        self.assertEqual (gff_empty.boxsize, None)
        self.assertEqual (gff_empty.resolution, None)
    
    def test_generalfactory_init(self):
        test_box = (float(10),float(20),float(30))
        test_res = [round(5),round(4),round(6)]
        gff = GeneralFieldFactory(test_box,test_res)
        # test common base class init
        self.assertEqual (gff.field_type, 1)
        self.assertEqual (gff.name, 'general')
        self.assertEqual (gff.field_class, GeneralField)
        self.assertEqual (gff.boxsize, test_box)
        self.assertEqual (gff.resolution, (round(5),round(4),round(6)))
        self.assertEqual (gff.active_parameters, ())
        self.assertEqual (gff.default_parameters, {})
        self.assertEqual (gff.parameter_ranges, {})
        self.assertEqual (gff.default_variables, {})

    def test_empty_testfactory_init(self):
        tff_empty = TestFieldFactory()
        # test empty factory
        self.assertEqual (tff_empty.boxsize, None)
        self.assertEqual (tff_empty.resolution, None)
    
    def test_testfactory_init(self):
        test_box = (float(10),float(20),float(30))
        test_res = [round(5),round(4),round(6)]
        tff = TestFieldFactory(test_box,test_res, tuple('a'))
        # test test factory
        self.assertEqual (tff.name, 'test')
        self.assertEqual (tff.field_class, TestField)
        self.assertEqual (tff.active_parameters, tuple('a'))
        self.assertEqual (tff.default_parameters, {'a': 6.0,
                                                   'b': 0.0})
        self.assertEqual (tff.parameter_ranges, {'a':(float(0),float(12)),
                                                 'b':(float(0),float(4.5))})
        self.assertEqual (tff.default_variables, {'a':float(6)/float(12),
                                                  'b':float(0)/float(4.5)})
        
        tmp_field = tff.generate({'a':0.1},4,23)
                          
        self.assertEqual (tmp_field.parameters, {'a':float(0.1)*float(12),
                                                 'b':float(0.0),
                                                 'random_seed':round(23)})
        self.assertEqual (tmp_field.ensemble_size, round(4))

    def test_testfactory_light_init(self):
        tff = TestFieldFactory(active_parameters=tuple('a'))
        # test init with only active parameters
        self.assertEqual (tff.name, 'test')
        self.assertEqual (tff.field_class, TestField)
        self.assertEqual (tff.active_parameters, tuple('a'))
        self.assertEqual (tff.default_parameters, {'a': 6.0,
                                                   'b': 0.0})
        self.assertEqual (tff.parameter_ranges, {'a':(float(0),float(12)),
                                                 'b':(float(0),float(4.5))})
        self.assertEqual (tff.default_variables, {'a':float(6)/float(12),
                                                  'b':float(0)/float(4.5)})
        
        tmp_field = tff.generate({'a':0.1},4,None)
                          
        self.assertEqual (tmp_field.parameters, {'a':float(0.1)*float(12),
                                                 'b':float(0.0),
                                                 'random_seed':int(0)})
        self.assertEqual (tmp_field.ensemble_size, round(4))

if __name__ == '__main__':
    unittest.main()

    
        
        
    
    
