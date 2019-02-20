'''
Observable hosts NIFTy5.Field with two domains' product,
which is designed for ensemble of simulated maps.

invoked in simulator wrapper and likelihoods
for distributing/manipulating simulated outputs

Likelihood class contains _strip_data method for
extracting ndarray from Field

members:
.field
    -- NIFTy5 Field object for storing observable ensemble
.stripped
    -- stripping out ndarray from ._field
    designed for accessing ._field
.ensemble_mean
    -- calculate mean through realisations of ensemble
    return a single ndarray
.append
    -- append new observable data with various form

#undeciphered Theo's legacy
_to/from_hdf5

'''

import numpy as np

from nifty5 import Field, RGSpace, DomainTuple

class Observable(object):

    '''
    domain
        -- a tuple of NIFTy5.domain objects, in type NIFTy5.DomainTuple
        we restrict domain contains two NIFTy5.domain objects
        and the first one should be NIFTy5.RGSpace for convenience in calculating ensemble mean
    val
        -- actual observable data in the correct shape defined by domain.shape

    i.e., for an Helpix Nside=128 observable in ensemble of 10 realisations
    the correct domain setting reads:
    domain = nifty5.DomainTuple.make((nifty5.RGSpace(shape=(10,)),nifty5.HPSpace(nside=128)))
    then the corresponding val should be in shape (10,12*128*128)
    '''
    def __init__(self, domain=None, val=None):
        self.domain = domain
        self._field = Field.from_global_data(self.domain,val) # recommended in NIFTy5

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        assert (len(domain) == 2)
        assert isinstance(domain[0], RGSpace)
        for d in domain: # restrict to 1D
            assert (len(d.shape), int(1))
        self._domain = domain
    
    @property
    def ensemble_mean(self):
        try:
            self._ensemble_mean
        except AttributeError:
            self._ensemble_mean = self._field.mean(spaces=0).to_global_data()
        finally:
            return self._ensemble_mean

    '''
    indirectly visit ._field
    '''
    @property
    def stripped(self):
        return self._field.to_global_data()

    '''
    should be able to append new data in type
    numpy array
    list/tuple
    NIFTy5.Field
    Observable

    since Field is read only, to append new data
    we need to strip ndarray out, extend, then update Field
    '''
    def append(self, new_data):
        assert isinstance(new_data,(np.ndarray,list,tuple,Field,Observable))
        if isinstance(new_data, Field): # from raw Field
            assert (new_data.domain[1] == self._domain[1]) # check domain
            new_cache = np.vstack([self._field.to_global_data(),new_data.to_global_data()])
            assert (new_cache.shape[0] == self._domain[0].shape[0] + new_data.domain[0].shape[0])
            assert (new_cache.shape[1] == self._domain[1].shape[0])
        elif isinstance(new_data, Observable): # from Observable
            assert (new_data.domain[1] == self._domain[1]) # check domain
            new_cache = np.vstack([self._field.to_global_data(),new_data.stripped])
            assert (new_cache.shape[0] == self._domain[0].shape[0] + new_data.domain[0].shape[0])
            assert (new_cache.shape[1] == self._domain[1].shape[0])
        elif isinstance(new_data, (np.ndarray,list,tuple)): # from list/tuple/ndarray
            if (len(np.shape(new_data)) == 1): # single row
                assert (len(new_data) == self._domain.shape[1])
                new_cache = np.vstack([self._field.to_global_data(),new_data])
                assert (new_cache.shape[0] == self._domain[0].shape[0] + int(1))
                assert (new_cache.shape[1] == self._domain[1].shape[0])
            else: # multi row
                assert (np.shape(new_data)[1] == self._domain.shape[1])
                new_cache = np.vstack([self._field.to_global_data(),new_data])
                assert (new_cache.shape[0] == self._domain[0].shape[0] + np.shape(new_data)[0])
                assert (new_cache.shape[1] == self._domain[1].shape[0])
        else:
            raise TypeError('shouldnt happen')
        # update new_cache to ._field
        new_domain = DomainTuple.make((RGSpace(shape=(new_cache.shape[0],)),self._domain[1]))
        self._field = Field.from_global_data(new_domain,new_cache)
            
        
