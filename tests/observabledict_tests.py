import unittest
import numpy as np

from nifty5 import Field, UnstructuredDomain, RGSpace, HPSpace, DomainTuple

from imagine.observables.observable import Observable
from imagine.observables.observable_dict import ObservableDict, Measurements, Simulations, Covariances, Masks


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

    def test_maskdict_append_array(self):
        msk = np.random.randint(0, 2, 48).reshape(1, 48)
        mskdict = Masks()
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        self.assertListEqual(list((mskdict[('test', 'nan', '2', 'nan')].to_global_data())[0]), list(msk[0]))
        mskdict.append(('test', 'nan', '48', 'nan'), msk, True)
        self.assertListEqual(list((mskdict[('test', 'nan', '48', 'nan')].to_global_data())[0]), list(msk[0]))

    def test_meadict_apply_mask(self):
        msk = np.array([0, 1, 0, 1, 1]).reshape(1, 5)
        mskdict = Masks()
        mskdict.append(('test', 'nan', '5', 'nan'), msk, True)
        arr = np.array([0., 1., 2., 3., 4.]).reshape(1, 5)
        meadict = Measurements()
        meadict.append(('test', 'nan', '5', 'nan'), arr, True)
        meadict.apply_mask(mskdict)
        self.assertListEqual(list((meadict[('test', 'nan', '3', 'nan')].to_global_data())[0]), [1., 3., 4.])
        # HEALPix map
        msk = np.random.randint(0, 2, 48).reshape(1, 48)
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        arr = np.random.rand(1, 48)
        meadict.append(('test', 'nan', '2', 'nan'), arr)
        pix_num = int(0)
        for i in msk[0]:
            if i == 1:
                pix_num += int(1)
        mskdict._archive.pop(('test', 'nan', '5', 'nan'), None)
        meadict.apply_mask(mskdict)
        self.assertTrue(('test', 'nan', str(pix_num), 'nan') in meadict.keys())

    def test_covdict_apply_mask(self):
        msk = np.array([0, 1, 0, 1, 1]).reshape(1, 5)
        mskdict = Masks()
        mskdict.append(('test', 'nan', '5', 'nan'), msk, True)
        arr = np.random.rand(5, 5)
        covdict = Covariances()
        covdict.append(('test', 'nan', '5', 'nan'), arr, True)
        covdict.apply_mask(mskdict)
        arr = np.delete(arr, [0, 2], 0)
        arr = np.delete(arr, [0, 2], 1)
        for i in range(arr.shape[0]):
            self.assertListEqual(list((covdict[('test', 'nan', '3', 'nan')].to_global_data())[i]), list(arr[i]))


if __name__ == '__main__':
    unittest.main()
