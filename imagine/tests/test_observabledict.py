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
        measuredict.append(name=('test', None, 3, None),
                           data=arr,
                           otype='plain')  # plain array
        local_arr = measuredict[('test', None, 3, None)].data
        if mpirank == 0:
            assert np.allclose(local_arr[0], arr[0])
        hrr = np.random.rand(1, 48)
        measuredict.append(name=('test', None, 2, None),
                           data=hrr,
                           otype='HEALPix')  # healpix array
        local_arr = measuredict[('test', None, 2, None)].data
        if mpirank == 0:
            assert np.allclose(local_arr[0], hrr[0])

    def test_measuredict_append_observable(self):
        hrr = np.random.rand(1, 48)
        obs1 = Observable(hrr, 'measured')
        measuredict = Measurements()
        measuredict.append(name=('test', None, 2, None),
                           data=obs1)  # healpix Observable
        assert np.allclose(measuredict[('test', None, 2, None)].data[0], hrr[0])
        arr = np.random.rand(1, 3)
        obs2 = Observable(arr, 'measured')
        measuredict.append(name=('test', None, 3, None),
                           data=obs2,
                           otype='plain')  # plain Observable
        assert np.allclose(measuredict[('test', None, 3, None)].data[0], arr[0])

    def test_simdict_append_array(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(name=('test', None, 3, None),
                       data=arr,
                       otype='plain')  # plain array
        assert simdict[('test', None, 3, None)].shape == (2*mpisize, 3)
        assert np.allclose(simdict[('test', None, 3, None)].data, arr)
        hrr = np.random.rand(3, 48)
        simdict.append(name=('test', None, 2, None),
                       data=hrr,
                       otype='HEALPix')  # healpix array
        assert simdict[('test', None, 2, None)].shape == (3*mpisize, 48)
        assert np.allclose(simdict[('test', None, 2, None)].data, hrr)

    def test_simdict_append_array_twice(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(name=('test', None, 3, None),
                       data=arr,
                       otype='plain')  # plain array
        assert simdict[('test', None, 3, None)].shape == (2*mpisize, 3)
        simdict.append(name=('test', None, 3, None),
                       data=arr,
                       otype='plain')  # plain array
        assert simdict[('test', None, 3, None)].shape == (4*mpisize, 3)

    def test_simdict_append_observable(self):
        hrr = np.random.rand(2, 48)
        obs1 = Observable(hrr, 'simulated')
        simdict = Simulations()
        simdict.append(name=('test', None, 2, None),
                       data=obs1,
                       otype='HEALPix')  # healpix Observable
        assert simdict[('test', None, 2, None)].shape == (2*mpisize, 48)
        assert np.allclose(simdict[('test', None, 2, None)].data, hrr)
        arr = np.random.rand(5, 3)
        obs2 = Observable(arr, 'simulated')
        simdict.append(name=('test', None, 3, None),
                       data=obs2,
                       otype='plain')  # plain Observable
        assert simdict[('test', None, 3, None)].shape == (5*mpisize, 3)
        assert np.allclose(simdict[('test', None, 3, None)].data, arr)

    def test_covdict_append_array(self):
        cov = np.random.rand(2, 2*mpisize)
        covdict = Covariances()
        covdict.append(name=('test', None, 2*mpisize, None),
                       data=cov)  # plain covariance
        assert covdict[('test', None, 2*mpisize, None)].shape == (2*mpisize, 2*mpisize)
        assert np.allclose(covdict[('test', None, 2*mpisize, None)].data, cov)
        cov = np.random.rand(12*mpisize, 12*mpisize*mpisize)
        covdict.append(name=('test', None, mpisize, None),
                       data=cov)  # healpix covariance
        assert covdict[('test', None, mpisize, None)].shape == (12*mpisize*mpisize, 12*mpisize*mpisize)
        assert np.allclose(covdict[('test', None, mpisize, None)].data, cov)

    def test_covdict_append_observable(self):
        cov = Observable(np.random.rand(2, 2*mpisize), 'covariance')
        covdict = Covariances()
        covdict.append(name=('test', None, 2*mpisize, None),
                       data=cov)  # plain covariance
        assert np.allclose(covdict[('test', None, 2*mpisize, None)].data, cov.data)
        cov = Observable(np.random.rand(12*mpisize, 12*mpisize*mpisize), 'covariance')
        covdict.append(name=('test', None, mpisize, None),
                       data=cov)  # healpix covariance
        assert np.allclose(covdict[('test', None, mpisize, None)].data, cov.data)

    def test_maskdict_append_array(self):
        msk = np.random.randint(0, 2, 48).reshape(1, -1)
        comm.Bcast(msk, root=0)
        mskdict = Masks()
        mskdict.append(name=('test', None, 2, None),
                       data=msk)
        local_msk = mskdict[('test', None, 2, None)].data
        assert np.allclose(local_msk[0], msk[0])
        mskdict.append(name=('test', None, 48, None),
                       data=msk)
        local_msk = mskdict[('test', None, 48, None)].data
        assert np.allclose(local_msk[0], msk[0])

    def test_meadict_apply_mask(self):
        msk = np.array([0, 1, 0, 1, 1]).reshape(1, 5)
        mskdict = Masks()
        comm.Bcast(msk, root=0)
        mskdict.append(name=('test', None, 5, None),
                       data=msk)
        arr = np.array([0., 1., 2., 3., 4.]).reshape(1, 5)
        meadict = Measurements()
        meadict.append(name=('test', None, 5, None),
                       data=arr,
                       otype='plain')
        meadict = mskdict(meadict)
        assert np.allclose(meadict[('test', None, 3, None)].data[0], [1, 3, 4])
        # HEALPix map
        msk = np.random.randint(0, 2, 48).reshape(1, 48)
        comm.Bcast(msk, root=0)
        mskdict.append(name=('test', None, 2, None),
                       data=msk)
        arr = np.random.rand(1, 48)
        meadict.append(name=('test', None, 2, None),
                       data=arr, otype='HEALPix')
        pix_num = msk.sum()
        meadict = mskdict(meadict)
        assert ('test', None, pix_num, None) in meadict.keys()

    def test_covdict_apply_mask(self):
        msk = np.random.randint(0, 2, 2*mpisize).reshape(1, -1)
        mskdict = Masks()
        comm.Bcast(msk, root=0)
        mskdict.append(name=('test', None, 2*mpisize, None),
                       data=msk, otype='plain')
        cov = np.random.rand(2, 2*mpisize)
        covdict = Covariances()
        covdict.append(name=('test', None, 2*mpisize, None),
                       data=cov)
        covdict = mskdict(covdict)
        pix_num = msk.sum()
        assert ('test', None, pix_num, None) in covdict.keys()
