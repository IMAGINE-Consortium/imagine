import unittest
import numpy as np

import mpi4py

from nifty5 import Field, DomainTuple, RGSpace
from imagine.observables.observable import Observable
from imagine.tools.masker import mask_obs, mask_cov
from imagine.tools.random_seed import seed_generator
from imagine.tools.covariance_estimator import oas_mcov

comm = mpi4py.MPI.COMM_WORLD
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

    def test_mask(self):
        msk_arr = np.array([0., 1., 0., 1., 1., 0.]).reshape(1, 6)
        obs_arr = np.random.rand(1, 6)
        cov_arr = (Field.from_global_data(DomainTuple.make(RGSpace(shape=(6, 6))),
                                          np.random.rand(6, 6))).local_data
        # mask by methods
        test_obs = mask_obs(obs_arr, msk_arr)
        test_cov = mask_cov(cov_arr, msk_arr)
        # mask manually
        fid_obs = np.hstack([obs_arr[0, 1], obs_arr[0, 3], obs_arr[0, 4]])
        # gather global cov
        fid_cov = np.zeros((6, 6))
        comm.Gather(cov_arr, fid_cov, root=0)
        # mask cov by hand
        fid_cov = np.delete(fid_cov, [0, 2, 5], 0)
        fid_cov = np.delete(fid_cov, [0, 2, 5], 1)
        comm.Bcast(fid_cov, root=0)
        # compare mask on matrix
        for i in test_cov:
            self.assertTrue(i in fid_cov)
        # compare mask on array
        self.assertListEqual(list(test_obs[0]), list(fid_obs))

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


if __name__ == '__main__':
    unittest.main()
