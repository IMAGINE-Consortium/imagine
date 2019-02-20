import unittest
import numpy as np

from nifty5 import Field, UnstructuredDomain, RGSpace, HPSpace, DomainTuple
from imagine.observables.observable import Observable
    
class TestObservalbe(unittest.TestCase):

    def test_init(self):
        # test __init__
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2)))
        val = np.random.rand(3,48)
        obs = Observable(dtuple,val)
        # test domain shape
        self.assertEqual (obs.domain, dtuple)
        # test function stripped
        raw = obs.stripped
        for i in range(len(val)):
            self.assertListEqual (list(raw[i]), list(val[i]))
        # test function ensemble_mean
        mean = obs.ensemble_mean
        val_mean = np.mean(val,axis=0)
        self.assertListEqual (list(mean), list(val_mean))

    def test_append_list(self):
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2)))
        val = np.random.rand(3,48)
        obs = Observable(dtuple,val)
        # test function append with list
        new_data = list(np.random.rand(48))
        obs.append(new_data)
        raw_obs = obs.stripped
        new_val = np.vstack([val,new_data])
        for i in range(len(val)):
            self.assertListEqual (list(raw_obs[i]), list(new_val[i]))

    def test_append_tuple(self):
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2)))
        val = np.random.rand(3,48)
        obs = Observable(dtuple,val)
        # test function append with tuple
        new_data = tuple(np.random.rand(48))
        obs.append(new_data)
        raw_obs = obs.stripped
        new_val = np.vstack([val,new_data])
        for i in range(len(val)):
            self.assertListEqual (list(raw_obs[i]), list(new_val[i]))

    def test_append_array(self):
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2)))
        val = np.random.rand(3,48)
        obs = Observable(dtuple,val)
        # test function append with 1d array
        new_data = np.random.rand(48)
        obs.append(new_data)
        raw_obs = obs.stripped
        new_val = np.vstack([val,new_data])
        for i in range(len(val)):
            self.assertListEqual (list(raw_obs[i]), list(new_val[i]))

    def test_append_ndarray(self):
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2)))
        val = np.random.rand(3,48)
        obs = Observable(dtuple,val)
        # test function append with nd array
        new_data = np.random.rand(6,48)
        obs.append(new_data)
        raw_obs = obs.stripped
        new_val = np.vstack([val,new_data])
        for i in range(len(val)):
            self.assertListEqual (list(raw_obs[i]), list(new_val[i]))

    def test_append_field(self):
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2)))
        val = np.random.rand(3,48)
        obs = Observable(dtuple,val)
        # test function append with Field
        dtuple = DomainTuple.make((RGSpace(shape=(2,)),HPSpace(nside=2)))
        new_data = np.random.rand(2,48)
        new_field = Field.from_global_data(dtuple,new_data)
        obs.append(new_field)
        raw_obs = obs.stripped
        new_val = np.vstack([val,new_data])
        for i in range(len(val)):
            self.assertListEqual (list(raw_obs[i]), list(new_val[i]))

    def test_append_observable(self):
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2)))
        val = np.random.rand(3,48)
        obs = Observable(dtuple,val)
        # test function append with Observable
        dtuple = DomainTuple.make((RGSpace(shape=(8,)),HPSpace(nside=2)))
        new_data = np.random.rand(8,48)
        new_obs = Observable(dtuple,new_data)
        obs.append(new_obs)
        raw_obs = obs.stripped
        new_val = np.vstack([val,new_data])
        for i in range(len(val)):
            self.assertListEqual (list(raw_obs[i]), list(new_val[i]))
           
if __name__ == '__main__':
    unittest.main()
