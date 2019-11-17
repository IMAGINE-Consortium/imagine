import unittest
import numpy as np
from mpi4py import MPI
from imagine.observables.observable import Observable


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class TestObservalbes(unittest.TestCase):
    
    def test_init_measure(self):
        arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'measured')
        self.assertEqual(test_obs.dtype, 'measured')
        self.assertListEqual(list(arr[0]), list(test_obs.data[0]))
        self.assertListEqual(list(arr[0]), list(test_obs.ensemble_mean[0]))
        
    def test_init_covariance(self):
        arr = np.random.rand(1,mpisize)
        test_obs = Observable(arr, 'covariance')
        self.assertEqual(test_obs.dtype, 'covariance')
        self.assertListEqual(list(arr[0]), list(test_obs.data[0]))
        self.assertEqual(test_obs.size, mpisize)

    def test_append_ndarray(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'simulated')
        brr = np.random.rand(1,128)
        test_obs.append(brr)
        fullrr = np.vstack([arr,brr])
        for i in range(fullrr.shape[0]):
            self.assertListEqual(list(fullrr[i]), list(test_obs.data[i]))
    
    def test_append_obs(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'simulated')
        if not mpirank:
            brr = np.random.rand(1,128)
        else:
            brr = np.random.rand(1,128)
        test_obs2 = Observable(brr, 'simulated')
        test_obs2.append(test_obs)
        fullrr = np.vstack([arr,brr])
        for i in range(fullrr.shape[0]):
            self.assertTrue(test_obs2.data[i] in fullrr)
    
    def test_append_twice(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'simulated')
        brr = np.random.rand(1,128)
        test_obs.append(brr)
        crr = np.random.rand(2,128)
        test_obs.append(crr)
        fullrr = np.vstack([arr, brr, crr])
        for i in range(fullrr.shape[0]):
            self.assertTrue(test_obs.data[i] in fullrr)
    
    def test_append_with_rewrite(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'simulated')
        test_obs.rw_flag = True
        brr = np.random.rand(1,128)
        test_obs.append(brr)
        self.assertListEqual(list(test_obs.data[0]), list(brr[0]))
        
    def test_append_after_rewrite(self):
        arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'measured')
        if not mpirank:
            brr = np.random.rand(2,128)
        else:
            brr = np.random.rand(1,128)
        test_obs.rw_flag = True
        test_obs.append(brr)
        crr = np.random.rand(1,128)
        test_obs.rw_flag = False
        test_obs.append(crr)
        fullrr = np.vstack([brr, crr])
        for i in range(fullrr.shape[0]):
            self.assertTrue(test_obs.data[i] in fullrr)

if __name__ == '__main__':
    unittest.main()
