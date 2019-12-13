import numpy as np
from mpi4py import MPI
from copy import deepcopy
import logging as log
from imagine.tools.mpi_helper import mpi_mean, mpi_shape, mpi_prosecutor
from imagine.tools.icy_decorator import icy

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

@icy
class Observable(object):
    """
    Observable class is designed for storing/manipulating distributed information.
    For the testing suits, please turn to "imagine/tests/observable_tests.py".

    Parameters
    ----------
    data : numpy.ndarray
        distributed/copied data
    dtype : str
        Data type, must be either: 'measured', 'simulated' or 'covariance'
    """
    def __init__(self, data=None, dtype=None):
        self.dtype = dtype
        self.data = data
        self.rw_flag = False

    @property
    def data(self):
        """
        Data stored in the local processor (`numpy.ndarray`, read-only).
        """
        return self._data

    @property
    def shape(self):
        """
        Shape of the global array, i.e. considering all processors
        (`numpy.ndarray`, read-only).
        """
        return mpi_shape(self._data)

    @property
    def global_data(self):
        """
        Data gathered from all processors (`numpy.ndarray`, read-only).
        """
        return mpi_global(self._data)

    @property
    def size(self):
        """
        Local data size (`int`, read-only)
        """
        return self._data.shape[1]

    @property
    def ensemble_mean(self):
        log.debug('@ observable::ensemble_mean')
        if (self._dtype == 'measured'):
            assert (self._data.shape[0] == 1)
            return self._data
        elif (self._dtype == 'simulated'):
            return mpi_mean(self._data)
        else:
            raise TypeError('unsupported data type')

    @property
    def rw_flag(self):
        """
        Rewriting flag, if true, append method will perform rewriting
        """
        return self._rw_flag

    @property
    def dtype(self):
        """
        Data type, can be either: 'measured', 'simulated' or 'covariance'
        """
        return self._dtype

    @data.setter
    def data(self, data):
        log.debug('@ observable::data')
        if data is None:
            self._data = None
        else:
            assert (len(data.shape) == 2)
            assert isinstance(data, np.ndarray)
            if (self._dtype == 'measured'):
                assert (data.shape[0] == 1)
            self._data = np.copy(data)
            if (self._dtype == 'covariance'):
                g_rows, g_cols = self.shape
                assert (g_rows == g_cols)

    @dtype.setter
    def dtype(self, dtype):
        if dtype is None:
            raise ValueError('dtype cannot be none')
        else:
            assert (dtype in ('measured', 'simulated', 'covariance'))
            self._dtype = deepcopy(dtype)

    @rw_flag.setter
    def rw_flag(self, rw_flag):
        assert (rw_flag in (True, False))
        self._rw_flag = deepcopy(rw_flag)

    def append(self, new_data):
        """
        appending new data happends only to SIMULATED dtype
        the new data to be appended should also be distributed
        which makes the appending operation naturally in parallel
        """
        log.debug('@ observable::append')
        assert isinstance(new_data, (np.ndarray, Observable))
        if isinstance(new_data, np.ndarray):
            mpi_prosecutor(new_data)
            if (self._rw_flag):  # rewriting
                self._data = np.copy(new_data)
            else:
                self._data = np.vstack([self._data, new_data])
        elif isinstance(new_data, Observable):
            if (self._rw_flag):
                self._data = np.copy(new_data.data)
            else:
                self._data = np.vstack([self._data, new_data.data])
