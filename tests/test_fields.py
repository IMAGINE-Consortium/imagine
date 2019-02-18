'''
check functions as much as possible in base classes
'''
import unittest
import imagine as im

class TestFeilds(unittest.TestCase):
    
    def test_generalfield_init(self):
        gf = im.GeneralField({},2.2)
        
        self.assertEqual (gf.simulator_checklist, {})
        self.assertEqual (gf.ensemble_size, int(2))
        self.assertEqual (gf.parameters, {})
    
    def test_testfield_init(self):
        test_list = {'a':float(3),'b':float(4.2)}
        tf = im.TestField(test_list,128.3,34.8)
        
        self.assertEqual (tf.simulator_checklist,
                          {'a': ('./Test/Regular/a','value'),
                           'random_seed': ('./Test/Random','seed'),
                           'b': ('./Test/Random/b','value')})
        self.assertEqual (tf.ensemble_size, int(128.3))
        self.assertEqual (tf.parameters,
                          {'a':float(3),
                           'b':float(4.2),
                           'random_seed':int(34.8)})
    
    def test_generalfieldfactory_init(self):
        test_box = (float(10),float(20),float(30))
        test_res = [int(5),int(4),int(6)]
        gff = im.GeneralFieldFactory(test_box,test_res)
        
        self.assertEqual (gff.field_type, 1)
        self.assertEqual (gff.name, 'general')
        self.assertEqual (gff.field_class, im.GeneralField)
        self.assertEqual (gff.boxsize, test_box)
        self.assertEqual (gff.resolution, (int(5),int(4),int(6)))
        self.assertEqual (gff.active_parameters, ())
        self.assertEqual (gff.default_parameters, {})
        self.assertEqual (gff.parameter_ranges, {})
        self.assertEqual (gff.default_variables, {})
    
    def test_testfieldfactory_init(self):
        test_box = (float(10),float(20),float(30))
        test_res = [int(5),int(4),int(6)]
        tff = im.TestFieldFactory(test_box,test_res,('a',))
        
        self.assertEqual (tff.name, 'test')
        self.assertEqual (tff.field_class, im.TestField)
        self.assertEqual (tff.active_parameters, ('a',))
        self.assertEqual (tff.default_parameters, {'a': 6.0,
                                                   'b': 2.0})
        self.assertEqual (tff.parameter_ranges, {'a':(float(0),float(12)),
                                                 'b':(float(0),float(4.5))})
        self.assertEqual (tff.default_variables, {'a':float(6)/float(12),
                                                  'b':float(2)/float(4.5)})
        
        tmp_field = tff.generate({'a':0.1},4,23)
                          
        self.assertEqual (tmp_field.parameters, {'a':float(0.1)*float(12),
                                                 'b':float(2.0),
                                                 'random_seed':int(23)})
        self.assertEqual (tmp_field.ensemble_size, int(4))

if __name__ == '__main__':
    unittest.main()

    
        
        
    
    
