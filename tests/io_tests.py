import unittest
import os
import numpy as np
from mpi4py import MPI
from imagine.tools.io_handler import io_handler


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class TestIO(unittest.TestCase):
    
    def test_io_copy(self):
        arr = np.random.rand(1,128)
        comm.Bcast(arr, root=0)
        test_io = io_handler()
        test_io.write_copy(arr, 'test_io_matrix.hdf5', 'test_group/test_dataset')
        # read back
        arr_check = test_io.read_copy(test_io.file_path, 'test_group/test_dataset')
        # consistency check
        for i in range(arr.shape[0]):
            self.assertListEqual(list(arr[i]), list(arr_check[i]))
        # clean up
        if not mpirank:
            os.remove(test_io.file_path)
    
    def test_io_dist(self):
        if not mpirank:
            arr = np.random.rand(3,128)
        else:
            arr = np.random.rand(2,128)
        test_io = io_handler()
        test_io.write_dist(arr, 'test_io_matrix.hdf5', 'test_group/test_dataset')
        # read back
        arr_check = test_io.read_dist(test_io.file_path, 'test_group/test_dataset')
        # consistency check
        for i in range(arr.shape[0]):
            self.assertListEqual(list(arr[i]), list(arr_check[i]))
        # clean up
        if not mpirank:
            os.remove(test_io.file_path)

if __name__ == '__main__':
    unittest.main()
