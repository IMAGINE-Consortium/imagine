"""
Observable hosts NIFTy5.Field with two domains' product,
which is designed for ensemble of simulated maps.

invoked in simulator wrapper and likelihoods
for distributing/manipulating simulated outputs

Likelihood class contains _strip_data method for
extracting ndarray from Field

members:
.field
    -- NIFTy5 Field object for storing observable ensemble
.domain
    -- NIFTy5 DomainTuple, defined globally
.rw_flag
    -- rewriting flag, if true, append method will perform rewriting
.to_global_data
    -- stripping out ndarray from ._field
    designed for accessing ._field
.append
    -- append new observable data in various form
"""

import numpy as np

import mpi4py

from nifty5 import Field, RGSpace, DomainTuple
from imagine.tools.icy_decorator import icy

comm = mpi4py.MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


@icy
class Observable(object):

    def __init__(self, domain=None, val=float(0)):
        """
        initialize Observable with locally defined ndarray
        
        :param domain: a tuple of NIFTy5.domain objects, in type NIFTy5.DomainTuple
        we restrict domain contains two NIFTy5.domain objects
        and the first one should be NIFTy5.RGSpace for convenience in calculating ensemble mean
        :param val: actual observable data in the correct shape defined by domain.shape

        i.e., for an Helpix Nside=128 observable in ensemble of 10 realisations
        the correct domain setting reads:
        domain = nifty5.DomainTuple.make((nifty5.RGSpace(shape=(10,)),nifty5.HPSpace(nside=128)))
        then the corresponding val should be in shape (10,12*128*128)
        """
        self.domain = domain
        self.rw_flag = False
        self.field = val  # should be after domain setter
        self._ensemble_mean = np.zeros((1, self._field.shape[1]))

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        """
        domain is defined globally
        domain.shape[0] should be either 1 (for storing measurement/observational data),
        or multiple mpisize (for storing simulated data)
        """
        assert (len(domain) == 2)
        assert isinstance(domain[0], RGSpace)
        # either global 1D, or multiple MPI size
        assert(domain.shape[0] % mpisize == 0 or domain.shape[0] == 1)
        for d in domain:  # restrict to 1D
            assert (len(d.shape) == int(1))
        self._domain = domain

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, val):
        """
        if domain with domain.sahpe[0] == 1, get val from master node
        otherwise, gather val from all nodes
        """
        if isinstance(val, float):  # empty case
            self._field = Field.full(self._domain, val)
            self._rw_flag = True
        elif isinstance(val, np.ndarray):
            # if domain.shape[0] == 1, no MPI distribution
            if self._domain.shape[0] == 1:
                self._field = Field.from_global_data(self.domain, val)
            else:
                self._field = Field.from_local_data(self.domain, val)
        else:
            raise TypeError('unsupported data type')

    @property
    def rw_flag(self):
        return self._rw_flag

    @rw_flag.setter
    def rw_flag(self, rw_flag):
        self._rw_flag = rw_flag
    
    @property
    def ensemble_mean(self):
        self._ensemble_mean = (self._field.mean(spaces=0).to_global_data()).reshape(1, self._field.shape[1])
        return self._ensemble_mean

    @property
    def shape(self):
        return self._field.shape

    @property
    def local_data(self):
        return self._field.local_data

    def to_global_data(self):
        """
        indirectly visit ._field
        dont make it a preperty in order to be aligned with
        the same method in NIFTy5.Field
        
        :return: numpy ndarray of observable content
        """
        return self._field.to_global_data()

    def append(self, new_data):
        """
        append new_data from all nodes
        if the observable is not distributed, self.shape[0] % mpisize != 0,
        it should be a measurement/observational data, in this case,
        there is no need to append/rewrite new data,
        nor it is convenient to do so
        
        :param new_data: new data in type numpy array, NIFTy5.Field, Observable
        :return:

        since Field is read only, to append new data
        we need to strip ndarray out, extend, then update Field
        append also handle ._flag is True case
        which means instead of append new data
        we should rewrite
        """
        assert(self._field.shape[0] % mpisize == 0)  # non-distributed, do not append
        # strip data
        if isinstance(new_data, (Field, Observable)):
            raw_new = new_data.local_data  # return to each node
        elif isinstance(new_data, np.ndarray):
            raw_new = new_data
        else:
            raise TypeError('unsupported type')
        # assemble new_cache
        if self.rw_flag:
            local_cache = raw_new
            self.rw_flag = False  # rw only once by default
        else:
            local_cache = np.vstack([self._field.local_data, raw_new])
        # update new_cache to ._field, first need to get global_size
        new_domain = DomainTuple.make((RGSpace(local_cache.shape[0]*mpisize), self._domain[1]))
        self._field = Field.from_local_data(new_domain, local_cache)
