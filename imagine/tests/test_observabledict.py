import unittest
import numpy as np
import mpi4py
from imagine.observables.observable_dict import Observable
from imagine.observables.observable_dict import ObservableDict, Measurements, Simulations, Covariances, Masks


comm = mpi4py.MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class TestObservableDicts(unittest.TestCase):
    
    def test_basedict(self):
        basedict = ObservableDict()
        self.assertEqual(basedict.archive, dict())
    
    def test_measuredict_append_array(self):
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        local_arr = measuredict[('test', 'nan', '3', 'nan')].data
        if mpirank == 0:
            self.assertListEqual(list(local_arr[0]), list(arr[0]))
        hrr = np.random.rand(1, 48)
        measuredict.append(('test', 'nan', '2', 'nan'), hrr)  # healpix array
        local_arr = measuredict[('test', 'nan', '2', 'nan')].data
        if mpirank == 0:
            self.assertListEqual(list(local_arr[0]), list(hrr[0]))
    
    def test_measuredict_append_observable(self):
        hrr = np.random.rand(1, 48)
        obs1 = Observable(hrr, 'measured')
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '2', 'nan'), obs1)  # healpix Observable
        self.assertListEqual(list((measuredict[('test', 'nan', '2', 'nan')].data)[0]), list(hrr[0]))
        arr = np.random.rand(1, 3)
        obs2 = Observable(arr, 'measured')
        measuredict.append(('test', 'nan', '3', 'nan'), obs2, True)  # plain Observable
        self.assertListEqual(list((measuredict[('test', 'nan', '3', 'nan')].data)[0]), list(arr[0]))
    
    def test_simdict_append_array(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        self.assertEqual(simdict[('test', 'nan', '3', 'nan')].shape, (2*mpisize, 3))
        for i in range(len(arr)):
            self.assertListEqual(list((simdict[('test', 'nan', '3', 'nan')].data)[i]), list(arr[i]))
        hrr = np.random.rand(3, 48)
        simdict.append(('test', 'nan', '2', 'nan'), hrr)  # healpix array
        self.assertEqual(simdict[('test', 'nan', '2', 'nan')].shape, (3*mpisize, 48))
        for i in range(len(hrr)):
            self.assertListEqual(list((simdict[('test', 'nan', '2', 'nan')].data)[i]), list(hrr[i]))
    
    def test_simdict_append_array_twice(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        self.assertEqual(simdict[('test', 'nan', '3', 'nan')].shape, (2*mpisize, 3))
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        self.assertEqual(simdict[('test', 'nan', '3', 'nan')].shape, (4*mpisize, 3))
    
    def test_simdict_append_observable(self):
        hrr = np.random.rand(2, 48)
        obs1 = Observable(hrr, 'simulated')
        simdict = Simulations()
        simdict.append(('test', 'nan', '2', 'nan'), obs1)  # healpix Observable
        self.assertEqual(simdict[('test', 'nan', '2', 'nan')].shape, (2*mpisize, 48))
        for i in range(len(hrr)):
            self.assertListEqual(list((simdict[('test', 'nan', '2', 'nan')].data)[i]), list(hrr[i]))
        arr = np.random.rand(5, 3)
        obs2 = Observable(arr, 'simulated')
        simdict.append(('test', 'nan', '3', 'nan'), obs2, True)  # plain Observable
        self.assertEqual(simdict[('test', 'nan', '3', 'nan')].shape, (5*mpisize, 3))
        for i in range(len(arr)):
            self.assertListEqual(list((simdict[('test', 'nan', '3', 'nan')].data)[i]), list(arr[i]))
    
    def test_covdict_append_array(self):
        cov = np.random.rand(2, 2*mpisize)
        covdict = Covariances()
        covdict.append(('test', 'nan', str(2*mpisize), 'nan'), cov, True)  # plain covariance
        self.assertEqual(covdict[('test', 'nan', str(2*mpisize), 'nan')].shape, (2*mpisize, 2*mpisize))
        for i in range(len(cov)):
            self.assertListEqual(list((covdict[('test', 'nan', str(2*mpisize), 'nan')].data)[i]), list(cov[i]))
        cov = np.random.rand(12*mpisize, 12*mpisize*mpisize)
        covdict.append(('test', 'nan', str(mpisize), 'nan'), cov)  # healpix covariance
        self.assertEqual(covdict[('test', 'nan', str(mpisize), 'nan')].shape, (12*mpisize*mpisize, 12*mpisize*mpisize))
        for i in range(len(cov)):
            self.assertListEqual(list((covdict[('test', 'nan', str(mpisize), 'nan')].data)[i]), list(cov[i]))
    
    def test_covdict_append_observable(self):
        cov = Observable(np.random.rand(2, 2*mpisize), 'covariance')
        covdict = Covariances()
        covdict.append(('test', 'nan', str(2*mpisize), 'nan'), cov, True)  # plain covariance
        for i in range(len(cov.data)):
            self.assertListEqual(list((covdict[('test', 'nan', str(2*mpisize), 'nan')].data)[i]), list((cov.data)[i]))
        cov = Observable(np.random.rand(12*mpisize, 12*mpisize*mpisize), 'covariance')
        covdict.append(('test', 'nan', str(mpisize), 'nan'), cov)  # healpix covariance
        for i in range(len(cov.data)):
            self.assertListEqual(list((covdict[('test', 'nan', str(mpisize), 'nan')].data)[i]), list((cov.data)[i]))
    
    def test_maskdict_append_array(self):
        msk = np.random.randint(0, 2, 48).reshape(1, -1)
        comm.Bcast(msk, root=0)
        mskdict = Masks()
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        local_msk = mskdict[('test', 'nan', '2', 'nan')].data
        self.assertListEqual(list(local_msk[0]), list(msk[0]))
        mskdict.append(('test', 'nan', '48', 'nan'), msk, True)
        local_msk = mskdict[('test', 'nan', '48', 'nan')].data
        self.assertListEqual(list(local_msk[0]), list(msk[0]))
    
    def test_meadict_apply_mask(self):
        msk = np.array([0, 1, 0, 1, 1]).reshape(1, 5)
        mskdict = Masks()
        comm.Bcast(msk, root=0)
        mskdict.append(('test', 'nan', '5', 'nan'), msk, True)
        arr = np.array([0., 1., 2., 3., 4.]).reshape(1, 5)
        meadict = Measurements()
        meadict.append(('test', 'nan', '5', 'nan'), arr, True)
        meadict.apply_mask(mskdict)
        self.assertListEqual(list((meadict[('test', 'nan', '3', 'nan')].data)[0]), [1., 3., 4.])
        # HEALPix map
        msk = np.random.randint(0, 2, 48).reshape(1, 48)
        comm.Bcast(msk, root=0)
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        arr = np.random.rand(1, 48)
        meadict.append(('test', 'nan', '2', 'nan'), arr)
        pix_num = msk.sum()
        meadict.apply_mask(mskdict)
        self.assertTrue(('test', 'nan', str(pix_num), 'nan') in meadict.keys())
    
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
        self.assertTrue(('test', 'nan', str(pix_num), 'nan') in covdict.keys())

if __name__ == '__main__':
    unittest.main()
