# %% IMPORTS
# Package imports
from mpi4py import MPI
import numpy as np
import os
import pytest
import scipy.sparse as spr

# IMAGINE imports
from imagine import rc
from imagine.tools import (
    empirical_cov, oas_cov, oas_mcov, mpi_mean, mpi_arrange, mpi_trans,
    mpi_mult, mpi_eye, mpi_trace, mpi_shape, mpi_lu_solve, mpi_slogdet,
    mpi_global, mpi_local, mask_obs, mask_cov, seed_generator, config, sparse)

# Globals
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

# Marks tests in this module as quick
pytestmark = pytest.mark.quick

# %% PYTEST DEFINITIONS
class TestTools(object):
    def test_seed(self):
        # test seed gen, in base class
        s1 = seed_generator(0)
        s2 = seed_generator(0)
        assert s1 != s2
        s3 = seed_generator(23)
        assert s3 == 23

    def test_shape(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_shape = mpi_shape(arr)
        test_shape[0] == mpisize+1
        test_shape[1] == 128

    def test_mean(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        full_arr = np.vstack(comm.allgather(arr))
        test_arr = (np.mean(full_arr, axis=0)).reshape(1,-1)
        test_mean = mpi_mean(arr)
        # check if almost equal since we forced the array datatype into numpy.float64
        assert np.allclose(test_mean[0], test_arr[0])

    def test_mask(self):
        msk_arr = np.random.choice([0,1], size=(1,128))
        msk_arr = comm.bcast(msk_arr, root=0)
        if not mpirank:
            dat_arr = np.random.rand(2,128)
        else:
            dat_arr = np.random.rand(1,128)
        cov_arr = np.random.rand(mpi_arrange(128)[1]-mpi_arrange(128)[0], 128)
        # mask by methods
        dat_msk = mask_obs(dat_arr, msk_arr)
        cov_msk = mask_cov(cov_arr, msk_arr)
        # mask manually
        test_dat = dat_arr*msk_arr
        test_dat = test_dat[test_dat != 0]
        dat_msk = dat_msk[dat_msk != 0]
        assert np.allclose(test_dat, dat_msk)
        #
        cov_mat = np.vstack(comm.allgather(cov_arr))
        cov_mat = cov_mat*msk_arr
        cov_mat = np.transpose(cov_mat)
        cov_mat = cov_mat*msk_arr
        cov_mat = np.transpose(cov_mat)
        cov_mat = cov_mat[cov_mat != 0]
        test_cov = np.vstack(comm.allgather(cov_msk))
        test_cov = test_cov[test_cov != 0]
        assert np.allclose(test_cov, test_cov)

    def test_trans(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_arr = mpi_trans(arr)
        full_arr = np.transpose(np.vstack(comm.allgather(arr)))
        local_begin, local_end = mpi_arrange(full_arr.shape[0])
        part_arr = full_arr[local_begin:local_end]
        assert np.allclose(part_arr, test_arr)

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
        assert np.allclose(part_c[0], test_c[0])

    def test_mpi_global(self):
        if not mpirank:
            arr_a = np.random.rand(2,128)
        else:
            arr_a = np.random.rand(1,128)
        full_a = np.vstack(comm.allgather(arr_a))
        test_a = mpi_global(arr_a)
        if not mpirank:
            full_a = full_a.reshape(1,-1)
            test_a = test_a.reshape(1,-1)
            assert np.allclose(full_a[0], test_a[0])

    def test_mpi_local(self):
        if not mpirank:
            arr_a = np.random.rand(32,128)
        else:
            arr_a = None
        test_a = mpi_local(arr_a)
        arr_a = comm.bcast(arr_a, root=0)
        local_a_begin, local_a_end = mpi_arrange(arr_a.shape[0])
        part_a = arr_a[local_a_begin:local_a_end,:]
        part_a = part_a.reshape(1,-1)
        test_a = test_a.reshape(1,-1)
        assert np.allclose(part_a[0], test_a[0])

    def test_mpi_eye(self):
        size = 128
        part_eye = mpi_eye(size)
        test_eye = np.eye(size, dtype=np.float64)
        full_eye = np.vstack(comm.allgather(part_eye))
        assert np.allclose(test_eye, full_eye)

    def test_mpi_trace(self):
        arr = np.random.rand(2,2*mpisize)
        test_trace = mpi_trace(arr)
        full_arr = np.vstack(comm.allgather(arr))
        true_trace = np.trace(full_arr)
        assert np.allclose(test_trace, true_trace)

    def test_empirical_cov(self):
        # mock observable ensemble with identical realizations
        arr = np.random.rand(1,32)
        comm.Bcast(arr, root=0)
        null_cov = np.zeros((32,32))
        # ensemble with identical realisations
        local_cov = empirical_cov(arr)
        full_cov = np.vstack(comm.allgather(local_cov))
        assert np.allclose(null_cov, full_cov)

    def test_oas_cov(self):
        # mock observable ensemble with identical realizations
        arr = np.random.rand(1,32)
        comm.Bcast(arr, root=0)
        null_cov = np.zeros((32,32))
        # ensemble with identical realisations
        local_cov = oas_cov(arr)
        full_cov = np.vstack(comm.allgather(local_cov))
        assert np.allclose(null_cov, full_cov)

    def test_oas_mcov(self):
        # mock observable ensemble with identical realizations
        arr = np.random.rand(1,32)
        comm.Bcast(arr, root=0)
        null_cov = np.zeros((32,32))
        # ensemble with identical realisations
        mean, local_cov = oas_mcov(arr)
        full_cov = np.vstack(comm.allgather(local_cov))
        assert np.allclose(mean[0], arr[0])
        assert np.allclose(null_cov, full_cov)

    def test_lu_solve(self):
        np.random.seed(mpirank)
        arr = np.random.rand(2, 2*mpisize)
        full_arr = np.vstack(comm.allgather(arr))
        brr = np.random.rand(1, 2*mpisize)
        comm.Bcast(brr, root=0)
        xrr = mpi_lu_solve(arr, brr)
        test_xrr = (np.linalg.solve(full_arr, brr.T)).T
        assert np.allclose(xrr[0], test_xrr[0])

    def test_lu_solve_odd(self):
        cols = 32
        rows = mpi_arrange(cols)[1] - mpi_arrange(cols)[0]
        arr = np.random.rand(rows, cols)
        full_arr = np.vstack(comm.allgather(arr))
        brr = np.random.rand(1, cols)
        comm.Bcast(brr, root=0)
        xrr = mpi_lu_solve(arr, brr)
        test_xrr = (np.linalg.solve(full_arr, brr.T)).T
        assert np.allclose(xrr[0], test_xrr[0])

    def test_slogdet(self):
        np.random.seed(mpirank)
        arr = np.random.rand(2, 2*mpisize)
        sign, logdet = mpi_slogdet(arr)
        full_arr = np.vstack(comm.allgather(arr))
        test_sign, test_logdet = np.linalg.slogdet(full_arr)
        assert sign == test_sign
        assert np.allclose(logdet, test_logdet)

    def test_slogdet_odd(self):
        cols = 32
        rows = mpi_arrange(cols)[1] - mpi_arrange(cols)[0]
        arr = np.random.rand(rows, cols)
        sign, logdet = mpi_slogdet(arr)
        full_arr = np.vstack(comm.allgather(arr))
        test_sign, test_logdet = np.linalg.slogdet(full_arr)
        assert sign == test_sign
        assert np.allclose(logdet, test_logdet)

    def test_slogdet_sparse(self):
        arr = np.random.random_sample(64).reshape(8,8)
        spr_arr = spr.csc_matrix(arr)
        sign, logdet = sparse.slogdet(spr_arr)
        sign_ref, logdet_ref = np.linalg.slogdet(arr)
        assert sign == sign_ref
        assert np.allclose(logdet, logdet_ref)

    def test_read_rc_from_env(self):
        # Tests whether the conversion of environment variables is working
        os.environ['IMAGINE_TEST_FLOAT'] = '3.14159'
        os.environ['IMAGINE_TEST_INT'] = '42'
        os.environ['IMAGINE_TEST_BOOL'] = 'T'
        os.environ['IMAGINE_TEST_BOOL2'] = 'False'

        # Includes new variables in rc for the test
        rc_test = {'test_float': 3.14159,
                   'test_int': 42,
                   'test_bool': True,
                   'test_bool2': False}

        rc.update({ k: None for k in rc_test.keys()})
        config.read_rc_from_env()

        for k, v in rc_test.items():
            assert v == rc[k]
            assert type(v) == type(rc[k])
