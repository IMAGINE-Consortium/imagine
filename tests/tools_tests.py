import unittest
import os
import numpy as np
from mpi4py import MPI

from imagine.tools.random_seed import seed_generator
from imagine.tools.mpi_helper import mpi_mean, mpi_arrange, mpi_trans
from imagine.tools.io_handler import io_handler
from imagine.tools.masker import mask_data, mask_cov
#from imagine.tools.covariance_estimator import empirical_cov
#from imagine import Observable


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


class TestTools(unittest.TestCase):

    def test_seed(self):
        # test seed gen, in base class
        s1 = seed_generator(0)
        s2 = seed_generator(0)
        self.assertNotEqual(s1, s2)
        s3 = seed_generator(48)
        self.assertEqual(s3, 48)
        
    def test_io_matrix(self):
        if not mpirank:
            arr = np.random.rand(3, 3)
        else:
            arr = np.random.rand(2, 3)
        test_io = io_handler()
        test_io.write(arr, 'test_io_matrix.hdf5', 'test_group/test_dataset')
        # read back
        arr_check = test_io.read(test_io.file_path, 'test_group/test_dataset')
        # consistency check
        for i in range(arr.shape[0]):
            self.assertListEqual(list(arr[i]), list(arr_check[i]))
        # cleanup
        if not mpirank:
            os.remove(test_io.file_path)
    
    def test_io_array(self):
        if not mpirank:
            arr = np.random.rand(1, 3)
        else:
            arr = np.random.rand(1, 3)
        test_io = io_handler()
        test_io.write(arr, 'test_io_matrix.hdf5', 'test_group/test_dataset')
        # read back
        arr_check = test_io.read(test_io.file_path, 'test_group/test_dataset')
        # consistency check
        self.assertListEqual(list(arr[0]), list(arr_check[0]))
        # cleanup
        if not mpirank:
            os.remove(test_io.file_path)
            
    def test_mean_array(self):
        arr = (np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])).reshape((1, 7))
        test_mean = mpi_mean(arr)
        self.assertListEqual(list(test_mean[0]), list(arr[0]))
        
    def test_mean_matrix(self):
        if not mpirank:
            arr = np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.1,0.2,0.3,0.4,0.5,0.6,0.7]])
        else:
            arr = (np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])).reshape((1,7))
        test_arr = (np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])).reshape((1,7))
        test_mean = mpi_mean(arr)
        # check if almost equal since we forced the array datatype into numpy.float64
        for i in range(len(test_mean[0])):
            self.assertAlmostEqual(test_mean[0][i], test_arr[0][i])
        
    def test_mask(self):
        msk_arr = np.array([0., 1., 0., 1., 1., 0.]).reshape(1, 6)
        dat_arr = np.random.rand(1, 6)
        cov_arr = np.random.rand(mpi_arrange(6)[1]-mpi_arrange(6)[0], 6)
        # mask by methods
        dat_msk = mask_data(dat_arr, msk_arr)
        cov_msk = mask_cov(cov_arr, msk_arr)
        # mask manually
        fid_dat = np.hstack([dat_arr[0, 1], dat_arr[0, 3], dat_arr[0, 4]])
        # gather global cov
        fid_cov = np.zeros((6, 6))
        comm.Gather(cov_arr, fid_cov, root=0)
        # mask cov by hand
        fid_cov = np.delete(fid_cov, [0, 2, 5], 0)
        fid_cov = np.delete(fid_cov, [0, 2, 5], 1)
        comm.Bcast(fid_cov, root=0)
        # compare mask on matrix
        for i in cov_msk:
            self.assertTrue(i in fid_cov)
        # compare mask on array
        self.assertListEqual(list(dat_msk[0]), list(fid_dat))
        
    def test_trans(self):
        if not mpirank:
            arr = np.random.rand(2, 5)
        else:
            arr = np.random.rand(1, 5)
        print (mpirank, arr)
        test_arr = mpi_trans(arr)
        print (mpirank, test_arr)
    
    '''
    def test_oas(self):
        # mock observable
        arr_a = np.random.rand(1, 4)
        comm.Bcast(arr_a, root=0)
        arr_ens = np.zeros((3, 4))
        null_cov = np.zeros((4, 4))
        # ensemble with identical realisations
        for i in range(len(arr_ens)):
            arr_ens[i] = arr_a
        dtuple = DomainTuple.make((RGSpace(3*mpisize), RGSpace(4)))
        obs = Observable(dtuple, arr_ens)
        test_mean, test_cov = oas_mcov(obs)
        for i in range(len(arr_a)):
            self.assertAlmostEqual(test_mean[0][i], arr_a[0][i])
            for j in range(len(arr_a)):
                self.assertAlmostEqual(test_cov[i][j], null_cov[i][j])
    '''
    
if __name__ == '__main__':
    unittest.main()
