import unittest
import numpy as np

import mpi4py

from nifty5 import Field, UnstructuredDomain, RGSpace, HPSpace, DomainTuple
from imagine import Observable

comm = mpi4py.MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


class TestObservalbes(unittest.TestCase):

    def test_init(self):
        # initialize observable
        dtuple = DomainTuple.make((RGSpace(3*mpisize), HPSpace(nside=2)))
        val = np.random.rand(3, 48)
        obs = Observable(dtuple, val)
        # collect local val at master
        val_master = None
        if mpirank == 0:
            val_master = np.zeros((3*mpisize, 48))
        comm.Gather(val, val_master, root=0)
        # test domain/field shape
        self.assertEqual(obs.domain, dtuple)
        if mpirank == 0:
            self.assertEqual(obs.shape, val_master.shape)
        # test function to_global_data()
        raw = obs.to_global_data()
        if mpirank == 0:
            for i in range(len(val_master)):
                self.assertListEqual(list(raw[i]), list(val_master[i]))
        # test function ensemble_mean
        mean = obs.ensemble_mean
        if mpirank == 0:
            val_mean = np.mean(val_master, axis=0)
            for i in range(val_mean.size):
                self.assertAlmostEqual(mean[0][i], val_mean[i])
    
    def test_1dinit(self):
        # initialize observable
        dtuple = DomainTuple.make((RGSpace(1), HPSpace(nside=2)))
        val = np.random.rand(1, 48)
        comm.Bcast(val, root=0)
        obs = Observable(dtuple, val)
        # matches val at master
        raw = obs.to_global_data()
        self.assertListEqual(list(raw[0]), list(val[0]))
    
    def test_append_1darray(self):
        # initialize observable
        dtuple = DomainTuple.make((RGSpace(3*mpisize), HPSpace(nside=2)))
        val = np.random.rand(3, 48)
        obs = Observable(dtuple, val)
        # test function append with 1d array
        new_data = np.random.rand(1, 48)
        obs.append(new_data)
        raw_obs = obs.to_global_data()
        new_val = np.vstack([val, new_data])
        # collect new val at master
        new_val_master = None
        if mpirank == 0:
            new_val_master = np.zeros((4*mpisize, 48))
        comm.Gather(new_val, new_val_master, root=0)
        # do comparison at master
        if mpirank == 0:
            for i in range(new_val_master.shape[0]):
                self.assertListEqual(list(raw_obs[i]), list(new_val_master[i]))
    
    def test_append_ndarray(self):
        dtuple = DomainTuple.make((RGSpace(3*mpisize), HPSpace(nside=2)))
        val = np.random.rand(3, 48)
        obs = Observable(dtuple, val)
        # test function append with nd array
        new_data = np.random.rand(6, 48)
        obs.append(new_data)
        raw_obs = obs.to_global_data()
        new_val = np.vstack([val, new_data])
        # collect new val at master
        new_val_master = None
        if mpirank == 0:
            new_val_master = np.zeros((9*mpisize, 48))
        comm.Gather(new_val, new_val_master, root=0)
        # do comparison at master
        if mpirank == 0:
            for i in range(new_val_master.shape[0]):
                self.assertListEqual(list(raw_obs[i]), list(new_val_master[i]))
    
    def test_append_twice(self):
        dtuple = DomainTuple.make((RGSpace(1*mpisize), HPSpace(nside=2)))
        val = np.random.rand(1, 48)
        obs = Observable(dtuple, val)
        # test function append with 1d array
        new_data = np.random.rand(1, 48)
        obs.append(new_data)
        self.assertEqual (obs.shape, (2*mpisize,48))
        obs.append(new_data)
        self.assertEqual (obs.shape, (3*mpisize,48))
    
    def test_append_field(self):
        dtuple = DomainTuple.make((RGSpace(3*mpisize), HPSpace(nside=2)))
        val = np.random.rand(3, 48)
        obs = Observable(dtuple, val)
        # test function append with Field
        dtuple = DomainTuple.make((RGSpace(2*mpisize), HPSpace(nside=2)))
        new_data = np.random.rand(2, 48)
        new_field = Field.from_local_data(dtuple, new_data)
        obs.append(new_field)
        raw_obs = obs.to_global_data()
        new_val = np.vstack([val, new_data])
        # collect new val at master
        new_val_master = None
        if mpirank == 0:
            new_val_master = np.zeros((5*mpisize, 48))
        comm.Gather(new_val, new_val_master, root=0)
        # do comparison at master
        if mpirank == 0:
            for i in range(new_val_master.shape[0]):
                self.assertListEqual(list(raw_obs[i]), list(new_val_master[i]))
    
    def test_append_observable(self):
        dtuple = DomainTuple.make((RGSpace(3*mpisize), HPSpace(nside=2)))
        val = np.random.rand(3, 48)
        obs = Observable(dtuple, val)
        # test function append with Observable
        dtuple = DomainTuple.make((RGSpace(5*mpisize), HPSpace(nside=2)))
        new_data = np.random.rand(5, 48)
        new_obs = Observable(dtuple, new_data)
        obs.append(new_obs)
        raw_obs = obs.to_global_data()
        new_val = np.vstack([val, new_data])
        # collect new val at master
        new_val_master = None
        if mpirank == 0:
            new_val_master = np.zeros((8*mpisize, 48))
        comm.Gather(new_val, new_val_master, root=0)
        # do comparison at master
        if mpirank == 0:
            for i in range(new_val_master.shape[0]):
                self.assertListEqual(list(raw_obs[i]), list(new_val_master[i]))
    
    def test_append_with_replace(self):
        dtuple = DomainTuple.make((RGSpace(1*mpisize), HPSpace(nside=2)))
        obs = Observable(dtuple)
        self.assertTrue(obs.rw_flag)
        # test with empty observable
        dtuple = DomainTuple.make((RGSpace(8*mpisize), HPSpace(nside=2)))
        new_data = np.random.rand(8, 48)
        new_obs = Observable(dtuple, new_data)
        obs.append(new_obs)
        raw_obs = obs.to_global_data()
        # collect new val at master
        new_val_master = None
        if mpirank == 0:
            new_val_master = np.zeros((8*mpisize, 48))
        comm.Gather(new_data, new_val_master, root=0)
        # do comparison at master
        if mpirank == 0:
            for i in range(new_val_master.shape[0]):
                self.assertListEqual(list(raw_obs[i]), list(new_val_master[i]))
    
    def test_append_after_replace(self):
        dtuple = DomainTuple.make((RGSpace(1*mpisize), HPSpace(nside=2)))
        obs = Observable(dtuple)
        self.assertTrue(obs.rw_flag)
        # test function append with 1d array
        new_data = np.random.rand(1, 48)
        obs.append(new_data)
        self.assertEqual (obs.shape, (1*mpisize,48))
        obs.append(new_data)
        self.assertEqual (obs.shape, (2*mpisize,48))


if __name__ == '__main__':
    unittest.main()
