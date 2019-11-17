import unittest
import numpy as np
from mpi4py import MPI

from imagine.tools.random_seed import seed_generator
from imagine.tools.mpi_helper import mpi_mean, mpi_arrange, mpi_trans, mpi_mult, mpi_eye, mpi_trace, mpi_shape
from imagine.tools.masker import mask_data, mask_cov
from imagine.tools.covariance_estimator import empirical_cov, oas_cov, oas_mcov


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class TestTools(unittest.TestCase):
    
    def test_seed(self):
        # test seed gen, in base class
        s1 = seed_generator(0)
        s2 = seed_generator(0)
        self.assertNotEqual(s1,s2)
        s3 = seed_generator(23)
        self.assertEqual(s3,23)

    def test_shape(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_shape = mpi_shape(arr)
        self.assertEqual(test_shape[0], mpisize+1)
        self.assertEqual(test_shape[1], 128)
    
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
            arr_a = np.random.rand(2,128)
        else:
            arr_a = np.random.rand(1,128)
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
                
    def test_mpi_eye(self):
        size = 128
        part_eye = mpi_eye(size)
        test_eye = np.eye(size, dtype=np.float64)
        full_eye = np.vstack(comm.allgather(part_eye))
        for i in range(full_eye.shape[0]):
            self.assertListEqual(list(test_eye[i]), list(full_eye[i]))
            
    def test_mpi_trace(self):
        arr = np.random.rand(2,2*mpisize)
        test_trace = mpi_trace(arr)
        full_arr = np.vstack(comm.allgather(arr))
        true_trace = np.trace(full_arr)
        self.assertAlmostEqual(test_trace, true_trace)
    
    def test_empirical_cov(self):
        # mock observable ensemble with identical realizations
        arr = np.random.rand(1,32)
        comm.Bcast(arr, root=0)
        null_cov = np.zeros((32,32))
        # ensemble with identical realisations
        local_cov = empirical_cov(arr)
        full_cov = np.vstack(comm.allgather(local_cov))
        for i in range(full_cov.shape[0]):
            for j in range(full_cov.shape[1]):
                self.assertAlmostEqual(null_cov[i][j], full_cov[i][j])
                
    def test_oas_cov(self):
        # mock observable ensemble with identical realizations
        arr = np.random.rand(1,32)
        comm.Bcast(arr, root=0)
        null_cov = np.zeros((32,32))
        # ensemble with identical realisations
        local_cov = oas_cov(arr)        
        full_cov = np.vstack(comm.allgather(local_cov))
        for i in range(full_cov.shape[0]):
            for j in range(full_cov.shape[1]):
                self.assertAlmostEqual(null_cov[i][j], full_cov[i][j])
                
    def test_oas_mcov(self):
        # mock observable ensemble with identical realizations
        arr = np.random.rand(1,32)
        comm.Bcast(arr, root=0)
        null_cov = np.zeros((32,32))
        # ensemble with identical realisations
        mean, local_cov = oas_mcov(arr)        
        full_cov = np.vstack(comm.allgather(local_cov))
        for k in range(mean.shape[1]):
            self.assertAlmostEqual(mean[0][k], arr[0][k])
        for i in range(full_cov.shape[0]):
            for j in range(full_cov.shape[1]):
                self.assertAlmostEqual(null_cov[i][j], full_cov[i][j])

if __name__ == '__main__':
    unittest.main()
