import unittest
import numpy as np
from nifty import Field, FieldArray
from imagine.observables.observable import Observable
from imagine.likelihoods.simple_likelihood import SimpleLikelihood

class TestLikeli(unittest.TestCase):

    def test_without_cov(self):
        arr_a = np.random.rand (3) # mock measurements
        measure = Field(val=0,
                        domain=FieldArray(shape=arr_a.shape),
                        distribution_strategy='equal')
        measure.val = arr_a
        l = SimpleLikelihood (['test'],{'test':measure},{'test':None})
        self.assertEqual (l.observable_names, ('test',))
        for i in range(len(arr_a)):
            self.assertEqual (l.measurements['test'][i], arr_a[i]) # check _strip_data
        arr_b = np.random.rand (2,3) # mock sims
        sims = Observable(val=0,
                          domain=(FieldArray(shape=(2,)),FieldArray(shape=(3,))),
                          distribution_strategy='equal')
        sims.val = arr_b
        rslt = l({'test':sims}) # calc by likelihood
        diff = (np.mean(arr_b,axis=0) - arr_a) # calc by hand
        baseline = -float(0.5)*float(np.vdot(diff,diff)+np.log(2*np.pi*3))
        self.assertEqual (rslt, baseline)
    
    def test_with_cov(self):
        arr_a = np.random.rand (3) # mock measurements
        measure = Field(val=0,
                        domain=FieldArray(shape=arr_a.shape),
                        distribution_strategy='equal')
        measure.val = arr_a
        arr_c = np.random.rand (3,3) # mock covariance
        cov = Field(val=0,
                    domain=FieldArray(shape=arr_c.shape),
                    distribution_strategy='equal')
        cov.val = arr_c
        l = SimpleLikelihood (['test'],{'test':measure},{'test':cov})
        self.assertEqual (l.observable_names, ('test',))
        for i in range(len(arr_a)):
            self.assertEqual (l.measurements['test'][i], arr_a[i]) # check _strip_data
        arr_b = np.random.rand (2,3) # mock sims
        sims = Observable(val=0,
                          domain=(FieldArray(shape=(2,)),FieldArray(shape=(3,))),
                          distribution_strategy='equal')
        sims.val = arr_b
        rslt = l({'test':sims}) # calc by likelihood
        diff = (np.mean(arr_b,axis=0) - arr_a) # calc by hand
        sign,logdet = np.linalg.slogdet(arr_c*2.*np.pi)
        baseline = -float(0.5)*float(np.vdot(diff,np.linalg.solve(arr_c,diff))+sign*logdet)
        self.assertEqual (rslt, baseline)


if __name__ == '__main__':
    unittest.main()
