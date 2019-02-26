import unittest
import numpy as np

from imagine.simulators.simulator import Simulator
from imagine.simulators.test.test_simulator import TestSimulator
from imagine.fields.test_field.test_field import TestField
from imagine.observables.observable_dict import Simulations, Measurements

class SimulatorTests(unittest.TestCase):

    def test_init(self):
        arr = np.random.rand(1,3)
        measuredict = Measurements()
        measuredict.append(('test','nan','3','nan'),arr,True)
        simer = TestSimulator(measuredict)
        self.assertEqual (len(simer.output_checklist), 1)
        self.assertEqual (simer.output_checklist, (('test','nan','3','nan'),))

    def test_generator(self):
        # mock measures
        arr = np.random.rand(1,3)
        measuredict = Measurements()
        measuredict.append(('test','nan','3','nan'),arr,True)
        # simulator
        simer = TestSimulator(measuredict)
        # test seed gen
        s1 = simer.seed_generator(0)
        s2 = simer.seed_generator(0)
        self.assertNotEqual (s1, s2)
        s3 = simer.seed_generator(48)
        self.assertEqual (s3, 48)
        # mock parameters, take short cut without fields
        mock_par = {'a':2.,'b':0.2,'random_seed':23}
        obs_arr = simer.obs_generator(mock_par,2,20)
        self.assertEqual (obs_arr.shape, (2,20))
        # test same random seed
        obs_arr_re = simer.obs_generator(mock_par,2,20)
        for i in range(obs_arr.shape[0]):
            self.assertListEqual (list(obs_arr[i]), list(obs_arr_re[i]))
        # test internal random seed
        mock_par = {'a':2.,'b':0.2,'random_seed':0}
        obs_arr_new = simer.obs_generator(mock_par,2,20)
        for i in range(obs_arr.shape[0]):
            for j in range(obs_arr.shape[1]):
                self.assertNotEqual (obs_arr[i][j], obs_arr_new[i][j])
        obs_arr_new_re = simer.obs_generator(mock_par,2,20)
        for i in range(obs_arr.shape[0]):
            for j in range(obs_arr.shape[1]):
                self.assertNotEqual (obs_arr_new[i][j], obs_arr_new_re[i][j])

    def test_generator_inout(self):
        # mock measures
        arr = np.random.rand(1,10)
        measuredict = Measurements()
        measuredict.append(('test','nan','10','nan'),arr,True)
        # mock field
        mock_par = {'a':2.,'b':0.2}
        # ensemble num and random seed injected here
        mock_field = TestField(mock_par,5,23)
        # simulator
        simer = TestSimulator(measuredict)
        # generating observable ensemble
        simdict = simer([mock_field])
        self.assertEqual (type(simdict), Simulations)
        self.assertEqual (len(simdict.keys()), 1)
        self.assertEqual (simdict[('test','nan','10','nan')].shape, (5,10))
        
        
if __name__ == '__main__':
    unittest.main()
