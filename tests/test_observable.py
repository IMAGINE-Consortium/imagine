import unittest
from nifty import Field, FieldArray, HPSpace
from imagine.observables.observable import Observable
import numpy as np

class TestNIFTy(unittest.TestCase):

    def test_array(self):
        # field with 2 by 3 array in each
        field = Field(val=1,
                      domain=FieldArray(shape=(2,3)),
                      distribution_strategy='equal')
        self.assertEqual (len(field.val), 2) # 2 rows
        self.assertEqual (len(field.val[0]),3) # 3 cols
        for i in field.val[0]:
            self.assertEqual (i,1.)
        inputlist = [2,6,21]
        field.val[1] = inputlist# reassign array
        for i in range(len(field.val[1])):
            self.assertEqual (field.val[1][i], inputlist[i])
    
    def test_healpix(self):
        # fieldarray & healpix fits
        temp = (FieldArray(shape=(3,)),HPSpace(nside=2)) # 3 healpix array with Nside 2
        field = Field(val=0,
                      domain=temp,
                      distribution_strategy='equal')
        self.assertEqual (len(field.val), 3) # 3 rows
        self.assertEqual (len(field.val[1]), 48)
    
class TestObservalbe(unittest.TestCase):
    
    def test_healpix_ensemblemean(self):
        domain = (FieldArray(shape=(2,)),HPSpace(nside=2))
        obs = Observable(val=0,
                         domain=domain,
                         distribution_strategy='equal')
        self.assertEqual (len(obs.val), 2) # 3 rows
        self.assertEqual (len(obs.val[1]), 48)
        pix = np.random.rand (2,48)
        obs.val = pix # input shape must fit in
        pix_mean = np.mean(pix,axis=0)
        obs_mean = obs.ensemble_mean().val.get_full_data()
        self.assertEqual (len(pix_mean),len(obs_mean))
        for i in range(len(pix_mean)):
            self.assertEqual (pix_mean[i],obs_mean[i])
           
if __name__ == '__main__':
    unittest.main()
