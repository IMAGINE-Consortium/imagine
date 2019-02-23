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
.to_global_data
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
import logging as log

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
    def __init__(self, domain=None, val=float(0)):
        self.domain = domain
        self.field = val # should be after domain setter

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        assert (len(domain) == 2)
        assert isinstance(domain[0], RGSpace)
        for d in domain: # restrict to 1D
            assert (len(d.shape) == int(1))
        self._domain = domain

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, val):
        self.rw_flag = False # re-write flag for empty case
        if isinstance(val, float): # empty case
            self._field = Field.full(self.domain,val)
            self.rw_flag = True
        else:
            # use np.vstack to reinforce correct shape
            self._field = Field.from_global_data(self.domain,np.vstack([val]))

    @property
    def rw_flag(self):
        return self._rw_flag

    @rw_flag.setter
    def rw_flag(self, rw_flag):
        self._rw_flag = rw_flag
        log.debug('set observable rewrite flag as %s' % str(rw_flag))
    
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
    def to_global_data(self):
        return self._field.to_global_data()

    '''
    should be able to append new data in type
    numpy array
    list/tuple
    NIFTy5.Field
    Observable

    since Field is read only, to append new data
    we need to strip ndarray out, extend, then update Field

    append also handle ._flag is True case
    which means instead of append new data
    we should rewrite
    '''
    def append(self, new_data):
        assert isinstance(new_data,(np.ndarray,list,tuple,Field,Observable))
        # strip data
        if isinstance(new_data, (Field,Observable)):
            try:
                raw_new = new_data.to_global_data() # from Field
            except AttributeError:
                raw_new = new_data.field.to_global_data() # from Observable
        elif isinstance(new_data, (np.ndarray,list,tuple)):
            raw_new = new_data # from list/tuple/ndarray
        else:
            raise TypeError('shouldnt happen')
        # assemble new_cache
        if self.rw_flag:
            new_cache = raw_new
            log.debug('new data rewrite to observable')
        else:
            new_cache = np.vstack([self._field.to_global_data(),raw_new])
            log.debug('new data append to observable')
        # update new_cache to ._field
        new_domain = DomainTuple.make((RGSpace(new_cache.shape[0]),self._domain[1]))
        self._field = Field.from_global_data(new_domain,new_cache)
