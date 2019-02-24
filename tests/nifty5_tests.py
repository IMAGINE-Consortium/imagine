'''
# nifty5 has dramatically changed from NIFTy3
# 1) Field values are read only
# 2) no distribution_strategy argument in Field
# 4) FieldArray is no longer supported
# I dont have time to try out all new features critical for IMAGINE
# just list a few key operations required:
# 1) allocating Field mem, distributed among computing nodes
# 2) pushing new arrays to Field, this comes from adding new realisation to simulated observable ensemble
# 3) calculate ensemble mean
# 4) write/read arrays with shape (ensemble_number,pixel_size) for simulated observable ensemble
# 5) write/read arrays with shape (pixel_size,pixel_size) for covariance matrices

directly construct Field is not recommended in NIFTy5
we do so for testing purpose
'''

import unittest
from nifty5 import Field, UnstructuredDomain, RGSpace, HPSpace, DomainTuple
import numpy as np

class NIFTyTests(unittest.TestCase):
    
    # test field domain shape
    def test_unstructured(self):
        # field with 2 by 3 array in each
        domain = DomainTuple.make(UnstructuredDomain(shape=(2,3)))
        field = Field(domain=domain,val=1.)
        self.assertEqual (len(field.val), 2) # 2 rows
        self.assertEqual (len(field.val[0]),3) # 3 cols

    # test field multidomain shape
    def test_unstructuredhealpix(self):
        # fieldarray & healpix fits
        dtuple = DomainTuple.make((UnstructuredDomain(shape=(3,)),HPSpace(nside=2))) # 3 healpix array with Nside 2
        field = Field(domain=dtuple,val=0)
        self.assertEqual (len(field.val), 3) # 3 rows
        self.assertEqual (len(field.val[1]), 48)

    # test field domain value assign
    def test_assignvalue(self):
        # nifty5 Field is read-only
        dtuple = DomainTuple.make(UnstructuredDomain(shape=(3,3)))
        rndarr = np.random.rand(3,3)
        field = Field(domain=dtuple, val=rndarr)
        for i in range(len(rndarr)):
            self.assertListEqual (list(field.val[i]), list(rndarr[i]))

    # test return domain value
    def test_retrivedata(self):
        # get full data from Field
        dtuple = DomainTuple.make(UnstructuredDomain(shape=(3,3)))
        rndarr = np.random.rand(3,3)
        field = Field(domain=dtuple, val=rndarr)
        fulldata = field.to_global_data() # returns a numpy.ndarray
        for i in range(len(rndarr)): # check values
            self.assertListEqual (list(fulldata[i]), list(rndarr[i]))

    # test add new array
    def test_addnewdata(self):
        # simple domain
        dtuple = DomainTuple.make(UnstructuredDomain(shape=(3,3)))
        rndarr = np.random.rand(3,3)
        field = Field(domain=dtuple, val=rndarr)
        fulldata = field.to_global_data() # returns a numpy.ndarray
        newarr = np.random.rand(3)
        rndarr = np.vstack([fulldata,newarr]) # add new row through numpy
        dtuple = DomainTuple.make(UnstructuredDomain(shape=rndarr.shape))
        field = Field.from_global_data(dtuple,rndarr)
        self.assertEqual (len(field.val), 4) # 4 rows
        self.assertEqual (len(field.val[0]),3) # 3 cols
        for i in range(len(rndarr)): # check values
            self.assertListEqual (list(field.val[i]), list(rndarr[i]))
        # multi domain
        dtuple = DomainTuple.make((UnstructuredDomain(shape=(3,)),HPSpace(nside=2))) # 3 healpix array with Nside 2
        rndarr = np.random.rand(3,48)
        field = Field(domain=dtuple, val=rndarr)
        fulldata = field.to_global_data() # returns a numpy.ndarray
        for i in range(len(rndarr)): # check values
            self.assertListEqual (list(fulldata[i]), list(rndarr[i]))
        newarr = np.random.rand(48)
        rndarr = np.vstack([fulldata,newarr]) # add new row through numpy
        dtuple = DomainTuple.make(UnstructuredDomain(shape=rndarr.shape))
        field = Field.from_global_data(dtuple,rndarr)
        self.assertEqual (len(field.val), 4) # 4 rows
        self.assertEqual (len(field.val[0]),48) # 3 cols
        for i in range(len(rndarr)): # check values
            self.assertListEqual (list(field.val[i]), list(rndarr[i]))

    # test mean over ensemble, works with default RGSpace
    def test_mean(self):
        # multi domain
        dtuple = DomainTuple.make((RGSpace(shape=(3,)),HPSpace(nside=2))) # 3 healpix array with Nside 2
        rndarr = np.random.rand(3,48)
        field = Field(domain=dtuple, val=rndarr)
        field_mean = field.mean(0) # average over UnstructuredDomain, return a new field
        test_mean = np.mean(rndarr,axis=0)
        self.assertListEqual (list(field_mean.val), list(test_mean))
        
    
if __name__ == '__main__':
    unittest.main()
