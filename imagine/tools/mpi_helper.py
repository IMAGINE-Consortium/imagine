"""
this mpi helper module is designed for parallel computing and data handling.
For the testing suits, please turn to "imagine/tests/tools_tests.py".
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

def mpi_arrange(size):
    """
    with known global size, number of mpi nodes, and current rank
    return the begin and end index for distributing the global size
    
    parameters
    ----------
    
    size
        integer
        the total size of target to be distributed
        it can be a row size or a column size
    
    return
    ------
    two integers in numpy.uint
    the begin and end index [begin,end] for slicing the target
    """
    res = min(mpirank, size%mpisize)
    ave = size//mpisize
    if (ave == 0):
        raise ValueError('over distribution')
    return np.uint(res + mpirank*ave), np.uint(res + (mpirank+1)*ave + int(mpirank < size%mpisize))

def mpi_prosecutor(data):
    """
    check if the data is distributed in the correct way
    covariance matrix is distributed exactly the same manner as multi-realization data
    if not, an error will be raised
    
    parameters
    ----------
    
    data
        numpy.ndarray
        the distributed data to be examined
    """
    assert isinstance(data, np.ndarray)
    # get the global shape
    local_rows = np.empty(mpisize, dtype=np.uint)
    local_cols = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
    comm.Allgather([np.array(data.shape[1], dtype=np.uint), MPI.LONG], [local_cols, MPI.LONG])
    check_begin, check_end = mpi_arrange(np.sum(local_rows))
    if (data.shape[0] != check_end - check_begin):
        raise ValueError('incorrect data allocation')
    if not np.any(local_cols-local_cols[0]):
        raise ValueError('incorrect data allocation')
        
def mpi_mean(data):
    """
    calculate the mean of distributed array
    prefers averaging along column direction
    but if given (1,n) data shape
    the average is done along row direction the result
    note that the numerical values will be converted into double
    
    parameters
    ----------
    
    data
        numpy.ndarray
        distributed data
        
    return
    ------
    numpy.ndarray
    copied data mean, which means the mean is copied to all nodes
    """
    assert (len(data.shape)==2)
    assert isinstance(data, np.ndarray)
    # get the global shape
    local_rows = np.empty(mpisize, dtype=np.uint)
    local_cols = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
    comm.Allgather([np.array(data.shape[1], dtype=np.uint), MPI.LONG], [local_cols, MPI.LONG])
    # do summation first before averaging out
    partial_sum = np.empty(data.shape[1], dtype=np.float64)
    partial_sum = np.sum(data, axis=0).reshape((data.shape[1],))
    total_sum = np.empty(data.shape[1], dtype=np.float64)
    comm.Allreduce ([partial_sum, MPI.DOUBLE], [total_sum, MPI.DOUBLE], op=MPI.SUM)
    avg = (total_sum / np.sum(local_rows)).reshape((1, data.shape[1]))
    return avg

def mpi_trans(data):
    """
    transpose distributed data
    note that the numerical values will be converted into double
    
    parameters
    ----------
    
    data
        numpy.ndarray
        distributed data
        
    return
    ------
    numpy.ndarray
    transposed data in distribution
    """
    assert (len(data.shape)==2)
    assert isinstance(data, np.ndarray)
    # get the global shape
    local_rows = np.empty(mpisize, dtype=np.uint)
    local_cols = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
    comm.Allgather([np.array(data.shape[1], dtype=np.uint), MPI.LONG], [local_cols, MPI.LONG])
    # the algorithm goes as cutting local data into sub-pieces
    # and passing them to corresponding nodes
    local_col_begin, local_col_end = mpi_arrange(local_cols[0])
    local_col_begins = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(local_col_begin, dtype=np.uint), MPI.LONG], [local_col_begins, MPI.LONG])
    # prepare empty slot
    new_data = np.empty((local_col_end-local_col_begin, np.sum(local_rows)), dtype=np.float64)
    #
    for target in range(mpisize):  # send to other ranks
        if (target != mpirank):
            local_sent_buf = np.transpose(data[:, local_col_begins[target]:local_col_begins[target+1]])
            comm.Send([local_sent_buf, MPI.DOUBLE], dest=target, tag=target)
    for source in range(mpisize):
        if (source != mpirank):  # recv from other ranks
            source_row_begin = local_rows[:source]
            source_row_end = local_rows[:source+1]
            comm.Recv([new_data[source_row_begin:source_row_end,:], MPI.DOUBLE], source=source, tag=mpirank)
        else:  # recv from self
            new_data[local_col_begin:local_col_end,:] = np.transpose(data[:,local_col_begin:local_col_end])
    return new_data
    
'''
def mpi_mult(data1, data2):
    """
    calculate matrix multiplication of two distributed data,
    the result is data1*data2 in multi-node distribution
    note that the numerical values will be converted into double
    
    parameters
    ----------
    
    data1
        numpy.ndarray
        distributed left side data
        
    data2
        numpy.ndarray
        distributed right side data
        
    return
    ------
    numpy.ndarray
    distributed
    """
    assert (len(data1.shape) == 2)
    assert (len(data2.shape) == 2)
    # get the global shape
    dat2_rows = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(data2.shape[0], dtype=np.uint), MPI.LONG], [dat2_rows, MPI.LONG])
'''
