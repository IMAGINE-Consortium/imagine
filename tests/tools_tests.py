import unittest
import os
import numpy as np
from mpi4py import MPI

from imagine.tools.random_seed import seed_generator
from imagine.tools.mpi_helper import mpi_mean, mpi_arrange, mpi_trans, mpi_mult
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
        
    def test_io(self):
        if not mpirank:
            arr = np.random.rand(3, 128)
        else:
            arr = np.random.rand(2, 128)
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
    
    def test_mean(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        full_arr = np.vstack(comm.allgather(arr))
        test_arr = (np.mean(full_arr, axis=0)).reshape(1,-1)
        test_mean = mpi_mean(arr)
        # check if almost equal since we forced the array datatype into numpy.float64
        for i in range(len(test_mean[0])):
            self.assertAlmostEqual(test_mean[0][i], test_arr[0][i])
    
    def test_mask(self):
        msk_arr = np.random.choice([0,1], size=(1,128))
        msk_arr = comm.bcast(msk_arr, root=0)
        if not mpirank:
            dat_arr = np.random.rand(2,128)
        else:
            dat_arr = np.random.rand(1,128)
        cov_arr = np.random.rand(mpi_arrange(128)[1]-mpi_arrange(128)[0], 128)
        # mask by methods
        dat_msk = mask_data(dat_arr, msk_arr)
        cov_msk = mask_cov(cov_arr, msk_arr)
        # mask manually
        test_dat = dat_arr*msk_arr
        test_dat = test_dat[test_dat != 0]
        dat_msk = dat_msk[dat_msk != 0]
        self.assertListEqual(list(test_dat), list(dat_msk))
        #
        cov_mat = np.vstack(comm.allgather(cov_arr))
        cov_mat = cov_mat*msk_arr
        cov_mat = np.transpose(cov_mat)
        cov_mat = cov_mat*msk_arr
        cov_mat = np.transpose(cov_mat)
        cov_mat = cov_mat[cov_mat != 0]
        test_cov = np.vstack(comm.allgather(cov_msk))
        test_cov = test_cov[test_cov != 0]
        self.assertListEqual(list(test_cov), list(test_cov))
    
    def test_trans(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_arr = mpi_trans(arr)
        full_arr = np.transpose(np.vstack(comm.allgather(arr)))
        local_begin, local_end = mpi_arrange(full_arr.shape[0])
        part_arr = full_arr[local_begin:local_end]
        for i in range(part_arr.shape[0]):
            self.assertListEqual(list(part_arr[i]), list(test_arr[i]))
    
    def test_mult(self):
        if not mpirank:
            arr_a = np.random.rand(2,32)
        else:
            arr_a = np.random.rand(1,32)
        arr_b = mpi_trans(arr_a)
        test_c = mpi_mult(arr_a, arr_b)
        # make comparison
        full_a = np.vstack(comm.allgather(arr_a))
        full_b = np.vstack(comm.allgather(arr_b))
        full_c = np.dot(full_a, full_b)
        local_begin, local_end = mpi_arrange(full_c.shape[0])
        part_c = (full_c[local_begin:local_end]).reshape(1,-1)
        test_c = test_c.reshape(1,-1)
        for i in range(len(part_c)):
            self.assertAlmostEqual(part_c[0][i], test_c[0][i])
            
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
