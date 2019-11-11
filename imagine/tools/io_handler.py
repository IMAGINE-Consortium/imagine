"""
The io_handler class is designed for IMAGINE I/O with HDF5+MPI support.
We expect distributed data shape on each node has the same column size,
but not necessarily the same row size.
writing to HDF5 gets each node working in parallel,
so the final data shape in the HDF5 is basically piling up all rows together.
For the testing suits, please turn to "imagine/tests/tools_tests.py".

attributes

._wk_dir
    working directory for I/O
    
._file_path
    the absolute path of the HDF5 binary file

member functions

.write
    write a hdf5 file

.read
    read a hdf5 file
"""

import numpy as np
from mpi4py import MPI
import h5py
import os
import logging as log
from imagine.tools.mpi_helper import mpi_arrange

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class io_handler(object):
    
    def __init__(self, wk_dir=None):
        """
        initialize with working directory
        
        parameters
        ----------
        
        wkdir
            string, the absolute path
        """
        if wk_dir is None:
            self.wk_dir = os.getcwd()
        else:
            self.wk_dir = wk_dir
        log.debug('set working directory at %s' % self._wk_dir)
        self.file_path = None
            
    @property
    def wk_dir(self):
        return self._wk_dir
    
    @property
    def file_path(self):
        return self._file_path
    
    @wk_dir.setter
    def wk_dir(self, wk_dir):
        assert isinstance(wk_dir, str)
        self._wk_dir = wk_dir
        
    @file_path.setter
    def file_path(self, file_path):
        if file_path is None:
            self._file_path = None
        else:
            assert isinstance(file_path, str)
            self._file_path = file_path
        
    def write(self, data, file, key):
        """
        write a distributed data-set into a binary file
        if the given filename does not exist then create one
        the data shape must be either in (m,n) on each node
        each node will write independently and parallel to the HDF5 file
        
        parameters
        ----------
        
        data
            numpy.ndarray, any datatype, distributed data
        
        file
            string, filename
          
        key
            string, in form 'group name/dataset name'
        """
        assert isinstance(data, np.ndarray)
        assert (len(data.shape) == 2)
        assert isinstance(file, str)
        assert isinstance(key, str)
        # combine wk_path with filename
        self.file_path = os.path.join(self._wk_dir, file)
        # write permission, create if not exist
        fh = h5py.File(self._file_path, 'w', driver='mpio', comm=comm)
        fh.atomic = True
        # know the global data shape and each rank's offset
        local_rows = np.empty(mpisize, dtype=np.uint)
        local_cols = np.empty(mpisize, dtype=np.uint)
        comm.Allgather([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
        comm.Allgather([np.array(data.shape[1], dtype=np.uint), MPI.LONG], [local_cols, MPI.LONG])
        # assert if cols are identical
        if np.any(local_cols-local_cols[0]):
            raise ValueError ('upsupported data shape')
        # set offset position for each node
        offset_begin = np.sum(local_rows[0:mpirank])
        offset_end = offset_begin + local_rows[mpirank]
        global_shape = (np.sum(local_rows), local_cols[0])
        # create group and dataset
        if not key in fh.keys():
            dset = fh.create_dataset(key, global_shape, maxshape=(None, None), dtype=data.dtype)
        else:  # rewrite
            dset = fh[key]
            dset.resize(global_shape)
        # fill data in parallel
        dset[offset_begin:offset_end,:] = data
        fh.close()
        
    def read(self, file, key):
        """
        read from a binary file
        and return a distributed data-set
        note that the binary file data should contain enough rows
        to be distributed on the available computing nodes
        otherwise the mpi_arrange function will raise an error
        
        parameters
        ----------
        
        data
            numpy.ndarray, distributed data
        
        file
            string, filename
          
        key
            string, in form 'group name/dataset name'
          
        return
        ------
        
        a distributed numpy.ndarray
        the output must be in either (1,n) or (m,n) shape on each node
        """
        assert isinstance(file, str)
        assert isinstance(key, str)
        # combine wk_path with filename
        self.file_path = os.path.join(self._wk_dir, file)
        # write permission, create if not exist
        fh = h5py.File(self._file_path, 'r', driver='mpio', comm=comm)
        fh.atomic = True
        global_shape = fh[key].shape
        offset_begin, offset_end = mpi_arrange(global_shape[0])
        data = fh[key][offset_begin:offset_end,:]
        return data
