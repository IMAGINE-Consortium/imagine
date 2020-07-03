# %% IMPORTS
# Package imports
from mpi4py import MPI
import numpy as np

# IMAGINE imports
from imagine.observables import Observable

# Globals
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


# %% PYTEST DEFINITIONS
class TestObservalbes(object):
    def test_init_measure(self):
        arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'measured')
        assert test_obs.dtype == 'measured'
        assert test_obs.shape == (mpisize, 128)
        assert np.allclose(arr[0], test_obs.data[0])
        assert np.allclose(arr[0], test_obs.ensemble_mean[0])

    def test_init_covariance(self):
        arr = np.random.rand(1,mpisize)
        test_obs = Observable(arr, 'covariance')
        assert test_obs.dtype ==  'covariance'
        assert test_obs.shape == (mpisize, mpisize)
        assert np.allclose(arr[0], test_obs.data[0])
        assert test_obs.size == mpisize

    def test_append_ndarray(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'simulated')
        brr = np.random.rand(1,128)
        test_obs.append(brr)
        global_shape = test_obs.shape
        globalrr = test_obs.global_data
        if not mpirank:
            assert global_shape == globalrr.shape
        fullrr = np.vstack([arr,brr])
        assert np.allclose(fullrr, test_obs.data)

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
        global_shape = test_obs2.shape
        globalrr = test_obs2.global_data
        if not mpirank:
            assert global_shape == globalrr.shape
        fullrr = np.vstack([arr,brr])
        assert np.alltrue(np.isin(test_obs2.data, fullrr))

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
        global_shape = test_obs.shape
        globalrr = test_obs.global_data
        if not mpirank:
            assert global_shape == globalrr.shape
        fullrr = np.vstack([arr, brr, crr])
        assert np.alltrue(np.isin(test_obs.data, fullrr))

    def test_append_with_rewrite(self):
        if not mpirank:
            arr = np.random.rand(2,128)
        else:
            arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'simulated')
        test_obs.rw_flag = True
        brr = np.random.rand(1,128)
        test_obs.append(brr)
        global_shape = test_obs.shape
        globalrr = test_obs.global_data
        if not mpirank:
            assert global_shape == globalrr.shape
        assert np.allclose(test_obs.data[0], brr[0])

    def test_append_after_rewrite(self):
        arr = np.random.rand(1,128)
        test_obs = Observable(arr, 'simulated')
        if not mpirank:
            brr = np.random.rand(2,128)
        else:
            brr = np.random.rand(1,128)
        test_obs.rw_flag = True
        test_obs.append(brr)
        crr = np.random.rand(1,128)
        # rw_flag must have be switched off
        test_obs.append(crr)
        global_shape = test_obs.shape
        globalrr = test_obs.global_data
        if not mpirank:
            assert global_shape == globalrr.shape
        fullrr = np.vstack([brr, crr])
        assert np.alltrue(np.isin(test_obs.data, fullrr))
