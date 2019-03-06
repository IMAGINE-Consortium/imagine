import unittest
import numpy as np

from nifty5 import Field, UnstructuredDomain, RGSpace, HPSpace, DomainTuple

from imagine.observables.observable import Observable
from imagine.observables.observable_dict import ObservableDict, Measurements, Simulations, Covariances


class TestObservableDicts(unittest.TestCase):

    def test_basedict(self):
        basedict = ObservableDict()
        self.assertEqual(basedict.archive, dict())
    
    def test_measuredict_append_array(self):
        arr = np.random.rand(1, 3)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        self.assertListEqual(list(measuredict[('test', 'nan', '3', 'nan')].to_global_data()[0]), list(arr[0]))
        hrr = np.random.rand(1, 48)
        measuredict.append(('test', 'nan', '2', 'nan'), hrr)  # healpix array
        self.assertListEqual(list(measuredict[('test', 'nan', '2', 'nan')].to_global_data()[0]), list(hrr[0]))

    def test_measuredict_append_observable(self):
        dtuple = DomainTuple.make((RGSpace(1), HPSpace(nside=2)))
        hrr = np.random.rand(1, 48)
        obs1 = Observable(dtuple, hrr)
        measuredict = Measurements()
        measuredict.append(('test', 'nan', '2', 'nan'), obs1)  # healpix Observable
        self.assertListEqual(list(measuredict[('test', 'nan', '2', 'nan')].to_global_data()[0]), list(hrr[0]))
        dtuple = DomainTuple.make((RGSpace(1), RGSpace(3)))
        arr = np.random.rand(1, 3)
        obs2 = Observable(dtuple, arr)
        measuredict.append(('test', 'nan', '3', 'nan'), obs2)  # plain Observable
        self.assertListEqual(list(measuredict[('test', 'nan', '3', 'nan')].to_global_data()[0]), list(arr[0]))
    
    def test_simdict_append_array(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        for i in range(len(arr)):
            self.assertListEqual(list((simdict[('test', 'nan', '3', 'nan')].to_global_data())[i]), list(arr[i]))
        hrr = np.random.rand(3, 48)
        simdict.append(('test', 'nan', '2', 'nan'), hrr)  # healpix array
        for i in range(len(hrr)):
            self.assertListEqual(list((simdict[('test', 'nan', '2', 'nan')].to_global_data())[i]), list(hrr[i]))

    def test_simdict_append_array_twice(self):
        arr = np.random.rand(2, 3)
        simdict = Simulations()
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        self.assertEqual(simdict[('test', 'nan', '3', 'nan')].shape, (2,3))
        simdict.append(('test', 'nan', '3', 'nan'), arr, True)  # plain array
        self.assertEqual(simdict[('test', 'nan', '3', 'nan')].shape, (4,3))
    
    def test_simdict_append_observable(self):
        dtuple = DomainTuple.make((RGSpace(2), HPSpace(nside=2)))
        hrr = np.random.rand(2, 48)
        obs1 = Observable(dtuple, hrr)
        simdict = Simulations()
        simdict.append(('test', 'nan', '2', 'nan'), obs1)  # healpix Observable
        for i in range(len(hrr)):
            self.assertListEqual(list((simdict[('test', 'nan', '2', 'nan')].to_global_data())[i]), list(hrr[i]))
        dtuple = DomainTuple.make((RGSpace(5), RGSpace(3)))
        arr = np.random.rand(5, 3)
        obs2 = Observable(dtuple, arr)
        simdict.append(('test', 'nan', '3', 'nan'), obs2, True)  # plain Observable
        for i in range(len(arr)):
            self.assertListEqual(list((simdict[('test', 'nan', '3', 'nan')].to_global_data())[i]), list(arr[i]))
    
    def test_covdict_append_array(self):
        cov = np.random.rand(5, 5)
        covdict = Covariances()
        covdict.append(('test', 'nan', '5', 'nan'), cov, True)  # plain covariance
        for i in range(len(cov)):
            self.assertListEqual(list((covdict[('test', 'nan', '5', 'nan')].to_global_data())[i]), list(cov[i]))
        cov = np.random.rand(48, 48)
        covdict.append(('test', 'nan', '2', 'nan'), cov)  # healpix covariance
        for i in range(len(cov)):
            self.assertListEqual(list((covdict[('test', 'nan', '2', 'nan')].to_global_data())[i]), list(cov[i]))
    
    def test_covdict_append_field(self):
        cov = np.random.rand(3, 3)
        dtuple = DomainTuple.make(RGSpace(shape=cov.shape))
        cov_field = Field.from_global_data(dtuple, cov)
        covdict = Covariances()
        covdict.append(('test', 'nan', '3', 'nan'), cov_field, True)  # plain covariance
        for i in range(len(cov)):
            self.assertListEqual(list((covdict[('test', 'nan', '3', 'nan')].to_global_data())[i]), list(cov[i]))
        cov = np.random.rand(48, 48)
        dtuple = DomainTuple.make(RGSpace(shape=cov.shape))
        cov_field = Field.from_global_data(dtuple, cov)
        covdict.append(('test', 'nan', '2', 'nan'), cov_field)  # healpix covariance
        for i in range(len(cov)):
            self.assertListEqual(list((covdict[('test', 'nan', '2', 'nan')].to_global_data())[i]), list(cov[i]))


if __name__ == '__main__':
    unittest.main()
