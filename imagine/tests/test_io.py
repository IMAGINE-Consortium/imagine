# %% IMPORTS
# Package imports
from mpi4py import MPI
import numpy as np
import pytest

# IMAGINE imports
from imagine.tools import IOHandler

# Globals
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

# Marks tests in this module as quick
pytestmark = pytest.mark.quick

# %% PYTEST DEFINITIONS
class TestIO(object):
    def test_io_copy(self):
        arr = np.random.rand(1,128)
        comm.Bcast(arr, root=0)
        test_io = IOHandler()
        test_io.write_copy(arr, 'test_io_matrix.hdf5', 'test_group/test_dataset')
        # read back
        arr_check = test_io.read_copy(test_io.file_path, 'test_group/test_dataset')
        # consistency check
        assert np.allclose(arr, arr_check)

    def test_io_dist(self):
        if not mpirank:
            arr = np.random.rand(3,128)
        else:
            arr = np.random.rand(2,128)
        test_io = IOHandler()
        test_io.write_dist(arr, 'test_io_matrix.hdf5', 'test_group/test_dataset')
        # read back
        arr_check = test_io.read_dist(test_io.file_path, 'test_group/test_dataset')
        # consistency check
        assert np.allclose(arr, arr_check)
