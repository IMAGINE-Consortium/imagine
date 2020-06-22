"""
The io_handler class is designed for IMAGINE I/O with HDF5+MPI,
but parallel HDF5 is not required.

There are two types of data reading,
corresponding to the data types defined in the Observable class.

    1. for reading 'measured' data (including mask maps),
    each node reads the full data.
    'read_copy' is designed for this case.

    2. for reading 'covariance' data,
    each node reads a certain rows.
    'read_dist' is designed for this case.

We do not require writing in parallel since the output workload is not heavy.
And there are also two types of data writing out, corresponds to the reading,
i.e., 'write_copy' and 'write_dist'.

For the testing suits, please turn to "imagine/tests/tools_tests.py".
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
    """
    Handles the I/O.

    Parameters
    ----------
    wkdir : string
        The absolute path of the working directory.
    """
    def __init__(self, wk_dir=None):
        if wk_dir is None:
            self.wk_dir = os.getcwd()
        else:
            self.wk_dir = wk_dir
        log.debug('set working directory at %s' % self._wk_dir)
        self.file_path = None

    @property
    def wk_dir(self):
        """
        String containing the absolute path of the working directory.
        """
        return self._wk_dir

    @property
    def file_path(self):
        """
        Absolute path of the HDF5 binary file.
        """
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
            
    def write_copy(self, data, file, key):
        """
        Writes a copied data-set into a HDF5 file.
        In practice, it writes out the data stored in the master node,
        by defaut taking all nodes have the same copies.
        
        Parameters
        ----------
        data : numpy.ndarray
            Distributed/copied data.
        file : str
            Strong for filename.
        key : str
            String for HDF5 group and dataset names, e.g., 'group name/dataset name'.
        """
        log.debug('@ io_handler::write')
        assert isinstance(data, np.ndarray)
        assert (len(data.shape) == 2)
        assert isinstance(file, str)
        assert isinstance(key, str)
        # combine wk_path with filename
        self.file_path = os.path.join(self._wk_dir, file)
        # master node writing
        if not mpirank:
            # write permission, create if not exist
            with h5py.File(self._file_path, mode='a') as fh:
                # create group and dataset
                if not key in fh.keys():
                    dset = fh.create_dataset(key, data.shape, maxshape=(None, None), dtype=data.dtype)
                else:  # rewrite
                    dset = fh[key]
                    dset.resize(data.shape)
                # write the master node piece
                dset[:,:] = data
        comm.Barrier()

    def write_dist(self, data, file, key):
        """
        Writes a distributed data-set into a HDF5 file.
        If the given filename does not exist then creates one
        the data shape must be either in (m,n) on each node,
        each node will pass its content to the master node
        who is in charge of sequential writing.

        Parameters
        ----------
        data : numpy.ndarray
            Distributed data.
        file : str
            String for filename.
        key : str
            String for HDF5 group and dataset names, e.g., 'group name/dataset name'.
        """
        log.debug('@ io_handler::write')
        assert isinstance(data, np.ndarray)
        assert (len(data.shape) == 2)
        assert isinstance(file, str)
        assert isinstance(key, str)
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
        # combine wk_path with filename
        self.file_path = os.path.join(self._wk_dir, file)
        # sequential writing
        if not mpirank:
            # write permission, create if not exist
            with h5py.File(self._file_path, mode='a') as fh:
                # create group and dataset
                if not key in fh.keys():
                    dset = fh.create_dataset(key, global_shape, maxshape=(None, None), dtype=data.dtype)
                else:  # rewrite
                    dset = fh[key]
                    dset.resize(global_shape)
                # write the master node piece
                dset[0:local_rows[0],:] = data
                for source in range(1,mpisize):
                    source_offset_begin = np.sum(local_rows[0:source])
                    source_offset_end = source_offset_begin + local_rows[source]
                    # receive data from slave nodes
                    source_data = comm.recv(source=source, tag=source)
                    # write to disk
                    dset[source_offset_begin:source_offset_end,:] = source_data
        else:  # else to ``if not mpirank``
            # send data to the master node
            comm.isend(data, dest=0, tag=mpirank)
        comm.Barrier()

    def read_copy(self, file, key):
        """
        Reads from a HDF5 file identically to all nodes,
        by doing so, each node contains an identical copy of the data stored
        in the file.
        
        Parameters
        ----------
        data : numpy.ndarray
            Distributed data.
        file : str
            String for filename.
        key : str
            String for HDF5 group and dataset names, e.g., 'group name/dataset name'.

        Returns
        -------
        Copied numpy.ndarray.
            The output must be in (1,n) shape on each node.
        """
        log.debug('@ io_handler::read_copied')
        assert isinstance(file, str)
        assert isinstance(key, str)
        # combine wk_path with filename
        self.file_path = os.path.join(self._wk_dir, file)
        # write permission, create if not exist
        with h5py.File(self._file_path, mode='r') as fh:
            assert (fh[key].shape[0] == 1)
            data = fh[key][:,:]
        comm.Barrier()
        return data
        
    def read_dist(self, file, key):
        """
        Reads from a HDF5 file and returns a distributed data-set.
        Note that the binary file data should contain enough rows
        to be distributed on the available computing nodes,
        otherwise the mpi_arrange function will raise an error.

        Parameters
        ----------
        data : numpy.ndarray
            Distributed data.
        file : str
            String for filename.
        key : str
            String for HDF5 group and dataset names, e.g., 'group name/dataset name'.

        Returns
        -------
        Distributed numpy.ndarray.
            The output must be in either at least (1,n), or (m,n) shape on each node.
        """
        log.debug('@ io_handler::read_dist')
        assert isinstance(file, str)
        assert isinstance(key, str)
        # combine wk_path with filename
        self.file_path = os.path.join(self._wk_dir, file)
        # write permission, create if not exist
        with h5py.File(self._file_path, mode='r') as fh:
            global_shape = fh[key].shape
            offset_begin, offset_end = mpi_arrange(global_shape[0])
            data = fh[key][offset_begin:offset_end,:]
        comm.Barrier()
        return data
