"""
In the Observable class we define three data types, i.e.,
- 'measured'
- 'simulated'
- 'covariance'
where
'measured' indicates the hosted data is from measurements,
which has a single realization,
'simulated' indicates the hosted data is from simulations,
which has multiple realizations,
'covariance' indicates the hosted data is a covariance matrix,
which has a single realization but by default should not be stored/read/written
by a single computing node.

'measred' data puts its identical copies on all computing nodes,
which means each node has a full storage of 'measured' data.

'simulated' data puts different realizations on different nodes,
which means each node has part of the full realizations,
but at least a full version of one single realization.

'covariance' data distributes itself into all computing nodes,
which means to have a full set of 'covariance' data,
we have to collect pieces from all the computing nodes.
"""

# %% IMPORTS
# Built-in imports
from copy import deepcopy
import logging as log

# Package imports
import astropy.units as u
import numpy as np

# IMAGINE imports
from imagine.tools import pmean, pshape, prosecutor, pglobal

# All declaration
__all__ = ['Observable']


# %% CLASS DEFINITIONS
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
    otype : str
        Observable type, must be either: 'HEALPix', 'tabular' or 'plain'
    """
    def __init__(self, data=None, dtype=None, coords=None, otype=None):
        self.dtype = dtype
        self.otype = otype

        if isinstance(data, u.Quantity):
            self.data = data.value
            self.unit = data.unit
        elif isinstance(data, np.ndarray):
            self.data = data
            self.unit = None
        else:
            raise ValueError

        self.coords = coords
        self.rw_flag = False

    @property
    def data(self):
        """
        Data stored in the LOCAL processor (`numpy.ndarray`, read-only).
        """
        return self._data

    @data.setter
    def data(self, data):
        """
        extra input format check for 'measured' and 'covariance'
        no extra check for 'simulated'
        """
        log.debug('@ observable::data')
        if data is None:
            self._data = None
        else:
            assert (len(data.shape) == 2)
            assert isinstance(data, np.ndarray)
            if (self._dtype == 'measured'):  # copy single-row data from memory
                assert (data.shape[0] == 1)
            self._data = np.copy(data)
            if (self._dtype == 'covariance'):
                assert np.equal(*self.shape)

    @property
    def shape(self):
        """
        Shape of the GLOBAL array, i.e. considering all processors
        (`numpy.ndarray`, read-only).
        """
        return pshape(self._data)  # estimate shape from all nodes

    @property
    def global_data(self):
        """
        Data gathered from ALL processors (`numpy.ndarray`, read-only).
        Note that only master node hosts the global data,
        while slave nodes hosts None.
        """
        return pglobal(self.data)

    @property
    def size(self):
        """
        Local data size (`int`, read-only)
        this size means the dimension of input data
        not the sample size of realizations
        """
        return self._data.shape[1]

    @property
    def ensemble_mean(self):
        log.debug('@ observable::ensemble_mean')
        if (self._dtype == 'measured'):
            assert (self._data.shape[0] == 1)  # single realization
            return self._data  # since each node has a full copy
        elif (self._dtype == 'simulated'):
            return pmean(self._data)  # calculate mean from all nodes
        else:
            raise TypeError('unsupported data type')

    @property
    def dtype(self):
        """
        Data type, can be either: 'measured', 'simulated' or 'covariance'
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype is None:
            raise ValueError('dtype cannot be None')
        else:
            assert (dtype in ('measured', 'simulated', 'covariance'))
            self._dtype = str(dtype)

    @property
    def rw_flag(self):
        """
        Rewriting flag, if true, append method will perform rewriting
        """
        return self._rw_flag

    @rw_flag.setter
    def rw_flag(self, rw_flag):
        assert isinstance(rw_flag, bool)
        self._rw_flag = rw_flag

    def append(self, new_data):
        """
        appending new data happens only to SIMULATED dtype
        the new data to be appended should also be distributed
        which makes the appending operation naturally in parallel

        rewrite flag will be switched off once rewriten has been performed
        """
        log.debug('@ observable::append')
        assert isinstance(new_data, (np.ndarray, Observable))
        assert (self._dtype == 'simulated')


        if isinstance(new_data, u.Quantity):
            assert self.unit == new_data.unit
            new_data = new_data.value

        if isinstance(new_data, np.ndarray):
            prosecutor(new_data)
            if (self._rw_flag):  # rewriting
                self._data = np.copy(new_data)
                self._rw_flag = False
            else:
                self._data = np.vstack([self._data, new_data])
        elif isinstance(new_data, Observable):
            if (self._rw_flag):
                self._data = np.copy(new_data.data)
                self._rw_flag = False
            else:
                self._data = np.vstack([self._data, new_data.data])
