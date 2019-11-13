"""
Observable class is designed for storing/manitulating distributed information

it is invoked in simulator wrapper and likelihoods
for distributing/manipulating simulated outputs

members:
.rw_flag
    -- rewriting flag, if true, append method will perform rewriting
.append
    -- append new observable data in various form
"""

import numpy as np
from mpi4py import MPI
from imagine.tools.mpi_helper import mpi_arrange, mpi_mean
from imagine.tools.icy_decorator import icy


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

@icy
class Observable(object):

    def __init__(self, shape=None):
        """
        initialize Observable with distributed numpy.ndarray
        
        parameters
        ----------
        
        shape
            a 2 element list/tuple which tells the
            total size of the distributed information
            e.g., for an Helpix Nside=128 observable in ensemble of N realisations
            then the corresponding shape is (N,12*128*128)
            e.g., for a Healpix Nside=128 observable in ensemble of single realization
            then the corresponding shape is (1,n) with n decided by the number of MPI ndoes
            
        val
            intial value for all element
        """
        self.shape = shape
        self.rw_flag = False
        self.ensemble_mean = self._shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        """
        shape is defined globally
        shape[0] should be either 
        1 (for storing measurement/observational data),
        or 
        multiple mpisize (for storing simulated data)
        
        the is a special requirement for multiple realizations,
        in order not to mess up with single realization data,
        we do not allow the number of realizations the same as
        MPI size, in which case the distribution would give (1,n) data
        shape for each MPI node, which is the same style as distributing
        a single realization data
        """
        assert (len(shape) == 2)
        assert isinstance(shape, (list,tuple))
        # either global 1D, or multiple MPI size
        assert(shape[0] % mpisize > 0 or shape[0] == int(1))
        self._shape = shape
        
    @property
    def data(self):
        return self._data
    
    @property
    def ensemble_mean(self):
        if self._shape[0] == 1:
            raise ValueError('ensemble mean works with ensembles')
        else:
            return mpi_mean(self._data)
    
    @ensemble_mean.setter
    def ensemble_mean(self, shape):
        """
        given the global data sample shape
        initialize the correct ensemble_mean array size with zeros
        """
        if (shape[0] == 1):
            self._ensemble_mean = np.zeros((1, mpi_arrange(shape[1], mpisize, mpirank)), dtype=np.double)
        else:
            self._ensemble_mean = np.zeros((mpi_arrange(shape[0], mpisize, mpirank), shape[1]), dtype=np.double)

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
