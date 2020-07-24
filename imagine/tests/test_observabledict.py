# %% IMPORTS
# Package imports
from mpi4py import MPI
import numpy as np
import pytest

# IMAGINE imports
from imagine.observables import (
    Observable, ObservableDict, Measurements, Simulations, Covariances, Masks)

# Globals
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

# Marks tests in this module as quick
pytestmark = pytest.mark.quick

# %% PYTEST DEFINITIONS
class TestObservableDicts(object):
    def test_basedict(self):
        with pytest.raises(TypeError):
            ObservableDict()

    def test_measuredict_append_array(self):
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        local_arr = measuredict[('test', 'nan', '3', 'nan')].data
        if mpirank == 0:
            assert np.allclose(local_arr[0], arr[0])
        hrr = np.random.rand(1, 48)
        measuredict.append(('test', 'nan', '2', 'nan'), hrr)  # healpix array
        local_arr = measuredict[('test', 'nan', '2', 'nan')].data
        if mpirank == 0:
            assert np.allclose(local_arr[0], hrr[0])

    def test_measuredict_append_observable(self):
        hrr = np.random.rand(1, 48)
        obs1 = Observable(hrr, 'measured')
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '2', 'nan'), obs1)  # healpix Observable
        assert np.allclose(measuredict[('test', 'nan', '2', 'nan')].data[0], hrr[0])
        arr = np.random.rand(1, 3)
        obs2 = Observable(arr, 'measured')
        measuredict.append(('test', 'nan', '3', 'nan'), obs2, True)  # plain Observable
        assert np.allclose(measuredict[('test', 'nan', '3', 'nan')].data[0], arr[0])

    def test_simdict_append_array(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        assert simdict[('test', 'nan', '3', 'nan')].shape == (2*mpisize, 3)
        assert np.allclose(simdict[('test', 'nan', '3', 'nan')].data, arr)
        hrr = np.random.rand(3, 48)
        simdict.append(('test', 'nan', '2', 'nan'), hrr)  # healpix array
        assert simdict[('test', 'nan', '2', 'nan')].shape == (3*mpisize, 48)
        assert np.allclose(simdict[('test', 'nan', '2', 'nan')].data, hrr)

    def test_simdict_append_array_twice(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        assert simdict[('test', 'nan', '3', 'nan')].shape == (2*mpisize, 3)
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        assert simdict[('test', 'nan', '3', 'nan')].shape == (4*mpisize, 3)

    def test_simdict_append_observable(self):
        hrr = np.random.rand(2, 48)
        obs1 = Observable(hrr, 'simulated')
        simdict = Simulations()
        simdict.append(('test', 'nan', '2', 'nan'), obs1)  # healpix Observable
        assert simdict[('test', 'nan', '2', 'nan')].shape == (2*mpisize, 48)
        assert np.allclose(simdict[('test', 'nan', '2', 'nan')].data, hrr)
        arr = np.random.rand(5, 3)
        obs2 = Observable(arr, 'simulated')
        simdict.append(('test', 'nan', '3', 'nan'), obs2, True)  # plain Observable
        assert simdict[('test', 'nan', '3', 'nan')].shape == (5*mpisize, 3)
        assert np.allclose(simdict[('test', 'nan', '3', 'nan')].data, arr)

    def test_covdict_append_array(self):
        cov = np.random.rand(2, 2*mpisize)
        covdict = Covariances()
        covdict.append(('test', 'nan', str(2*mpisize), 'nan'), cov, True)  # plain covariance
        assert covdict[('test', 'nan', str(2*mpisize), 'nan')].shape == (2*mpisize, 2*mpisize)
        assert np.allclose(covdict[('test', 'nan', str(2*mpisize), 'nan')].data, cov)
        cov = np.random.rand(12*mpisize, 12*mpisize*mpisize)
        covdict.append(('test', 'nan', str(mpisize), 'nan'), cov)  # healpix covariance
        assert covdict[('test', 'nan', str(mpisize), 'nan')].shape == (12*mpisize*mpisize, 12*mpisize*mpisize)
        assert np.allclose(covdict[('test', 'nan', str(mpisize), 'nan')].data, cov)

    def test_covdict_append_observable(self):
        cov = Observable(np.random.rand(2, 2*mpisize), 'covariance')
        covdict = Covariances()
        covdict.append(('test', 'nan', str(2*mpisize), 'nan'), cov, True)  # plain covariance
        assert np.allclose(covdict[('test', 'nan', str(2*mpisize), 'nan')].data, cov.data)
        cov = Observable(np.random.rand(12*mpisize, 12*mpisize*mpisize), 'covariance')
        covdict.append(('test', 'nan', str(mpisize), 'nan'), cov)  # healpix covariance
        assert np.allclose(covdict[('test', 'nan', str(mpisize), 'nan')].data, cov.data)

    def test_maskdict_append_array(self):
        msk = np.random.randint(0, 2, 48).reshape(1, -1)
        comm.Bcast(msk, root=0)
        mskdict = Masks()
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        local_msk = mskdict[('test', 'nan', '2', 'nan')].data
        assert np.allclose(local_msk[0], msk[0])
        mskdict.append(('test', 'nan', '48', 'nan'), msk, True)
        local_msk = mskdict[('test', 'nan', '48', 'nan')].data
        assert np.allclose(local_msk[0], msk[0])

    def test_meadict_apply_mask(self):
        msk = np.array([0, 1, 0, 1, 1]).reshape(1, 5)
        mskdict = Masks()
        comm.Bcast(msk, root=0)
        mskdict.append(('test', 'nan', '5', 'nan'), msk, True)
        arr = np.array([0., 1., 2., 3., 4.]).reshape(1, 5)
        meadict = Measurements()
        meadict.append(('test', 'nan', '5', 'nan'), arr, True)
        meadict.apply_mask(mskdict)
        assert np.allclose(meadict[('test', 'nan', '3', 'nan')].data[0], [1, 3, 4])
        # HEALPix map
        msk = np.random.randint(0, 2, 48).reshape(1, 48)
        comm.Bcast(msk, root=0)
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        arr = np.random.rand(1, 48)
        meadict.append(('test', 'nan', '2', 'nan'), arr)
        pix_num = msk.sum()
        meadict.apply_mask(mskdict)
        assert ('test', 'nan', str(pix_num), 'nan') in meadict.keys()

    def test_covdict_apply_mask(self):
        msk = np.random.randint(0, 2, 2*mpisize).reshape(1, -1)
        mskdict = Masks()
        comm.Bcast(msk, root=0)
        mskdict.append(('test', 'nan', str(2*mpisize), 'nan'), msk, True)
        cov = np.random.rand(2, 2*mpisize)
        covdict = Covariances()
        covdict.append(('test', 'nan', str(2*mpisize), 'nan'), cov, True)
        covdict.apply_mask(mskdict)
        pix_num = msk.sum()
        assert ('test', 'nan', str(pix_num), 'nan') in covdict.keys()
