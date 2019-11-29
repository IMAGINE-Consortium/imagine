import unittest
import os
import numpy as np
from mpi4py import MPI
from imagine.tools.io_handler import io_handler


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

class TestIO(unittest.TestCase):
    
    def test_io(self):
        if not mpirank:
            arr = np.random.rand(3,128)
        else:
            arr = np.random.rand(2,128)
        test_io = io_handler()
        test_io.write(arr, 'test_io_matrix.hdf5', 'test_group/test_dataset')
        # read back
        arr_check = test_io.read(test_io.file_path, 'test_group/test_dataset')
        # consistency check
        for i in range(arr.shape[0]):
            self.assertListEqual(list(arr[i]), list(arr_check[i]))
        # cleanup
        if not mpirank:
            os.remove(test_io.file_path)


if __name__ == '__main__':
    unittest.main()
