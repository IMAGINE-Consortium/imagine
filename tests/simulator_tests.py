import unittest
import numpy as np

import mpi4py

from imagine import Simulator
from imagine import LiSimulator
from imagine import BiSimulator
from imagine import TestField
from imagine import Simulations, Measurements

comm = mpi4py.MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


class TestSimulators(unittest.TestCase):

    def test_init(self):
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)
        simer = LiSimulator(measuredict)
        self.assertEqual(len(simer.output_checklist), 1)
        self.assertEqual(simer.output_checklist, (('test', 'nan', '3', 'nan'),))
        simer = BiSimulator(measuredict)
        self.assertEqual(len(simer.output_checklist), 1)
        self.assertEqual(simer.output_checklist, (('test', 'nan', '3', 'nan'),))

    def test_li(self):
        # mock measures
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)
        # simulator
        simer = LiSimulator(measuredict)
        # regular field check
        mock_par = {'a': 2, 'b': 0}
        field_list = [TestField(mock_par, 1, None)]
        obs_arr = simer.obs_generator(field_list, 1, 20)
        # calc by hand
        x = np.linspace(0., 2.*np.pi, 20)
        baseline = 2.*np.cos(x)
        self.assertListEqual(list(obs_arr[0]), list(baseline))
        # mock parameters, take short cut without fields
        mock_par = {'a': 2., 'b': 0.2}
        field_list = [TestField(mock_par, 2, [23]*2)]
        obs_arr = simer.obs_generator(field_list, 2, 20)
        self.assertEqual(obs_arr.shape, (2, 20))
        # test same random seed
        obs_arr_re = simer.obs_generator(field_list, 2, 20)
        for i in range(obs_arr.shape[0]):
            self.assertListEqual(list(obs_arr[i]), list(obs_arr_re[i]))
    
    def test_bi(self):
        # mock measures
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)
        # simulator
        simer = BiSimulator(measuredict)
        # regular field check
        mock_par = {'a': 2, 'b': 0}
        field_list = [TestField(mock_par, 1, None)]
        obs_arr = simer.obs_generator(field_list, 1, 20)
        # calc by hand
        x = np.linspace(0., 2.*np.pi, 20)
        baseline = np.square(2.*np.sin(x))
        self.assertListEqual(list(obs_arr[0]), list(baseline))
        # mock parameters, take short cut without fields
        mock_par = {'a': 2., 'b': 0.2}
        field_list = [TestField(mock_par, 2, [23]*2)]
        obs_arr = simer.obs_generator(field_list, 2, 20)
        self.assertEqual(obs_arr.shape, (2, 20))
        # test same random seed
        obs_arr_re = simer.obs_generator(field_list, 2, 20)
        for i in range(obs_arr.shape[0]):
            self.assertListEqual(list(obs_arr[i]), list(obs_arr_re[i]))

    def test_generator_inout(self):
        # mock measures
        arr = np.random.rand(1, 10)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '10', 'nan'), arr, True)
        # mock field
        mock_par = {'a': 2., 'b': 0.2}
        # ensemble num and random seed injected here
        np.random.seed(mpirank)  # assign different seed
        ensemble_seeds = np.random.randint(1, 10, 5)  # numpy random gives the same sequence by default on all nodes
        mock_field = TestField(mock_par, 5, ensemble_seeds)
        # simulator
        simer = LiSimulator(measuredict)
        # generating observable ensemble
        simdict = simer([mock_field])  # automatically append from all nodes
        self.assertEqual(type(simdict), Simulations)
        self.assertEqual(len(simdict.keys()), 1)
        self.assertEqual(simdict[('test', 'nan', '10', 'nan')].shape, (5*mpisize, 10))
        # simulator
        simer = BiSimulator(measuredict)
        # generating observable ensemble
        simdict = simer([mock_field])  # automatically append from all nodes
        self.assertEqual(type(simdict), Simulations)
        self.assertEqual(len(simdict.keys()), 1)
        self.assertEqual(simdict[('test', 'nan', '10', 'nan')].shape, (5*mpisize, 10))


if __name__ == '__main__':
    unittest.main()
