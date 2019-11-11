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
    # get the global shape before transpose
    local_rows = np.empty(mpisize, dtype=np.uint)
    local_cols = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
    comm.Allgather([np.array(data.shape[1], dtype=np.uint), MPI.LONG], [local_cols, MPI.LONG])
    # the algorithm cuts local data into sub-pieces and passing them to the corresponding nodes
    # which means we need to calculate the arrangement of pre-trans "columns" into post-trans "rows" 
    cut_col_begin, cut_col_end = mpi_arrange(local_cols[0])
    cut_col_begins = np.empty(mpisize, dtype=np.uint)
    cut_col_ends = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(cut_col_begin, dtype=np.uint), MPI.LONG], [cut_col_begins, MPI.LONG])
    comm.Allgather([np.array(cut_col_end, dtype=np.uint), MPI.LONG], [cut_col_ends, MPI.LONG])
    # prepare empty post-trans local data shape
    new_data = np.empty((cut_col_end-cut_col_begin, np.sum(local_rows)), dtype=np.float64)
    # send and receive sub-pieces
    for target in range(mpisize):
        if (target != mpirank):  # send to other ranks
            # the np.array(..., dtype=np.float64) is to ensure memory contiguous and data type correct
            # the np.transpose won't work, but changing the memory order will (cross-node copy)
            local_sent_buf = np.array(data[:, cut_col_begins[target]:cut_col_ends[target]], dtype=np.float64, order='F')
            comm.Send([local_sent_buf, MPI.DOUBLE], dest=target, tag=target)
        else:  # recv from self
            # note that transpose works here (in-node copy)
            new_col_begin = np.sum(local_rows[:mpirank])
            new_col_end = new_col_begin + local_rows[mpirank]
            new_data[:, new_col_begin:new_col_end] = np.transpose((data[:, cut_col_begin:cut_col_end]).astype(np.float64))
    for source in range(mpisize):
        if (source != mpirank):  # recv from other ranks
            local_recv_buf = np.empty((cut_col_ends[mpirank]-cut_col_begins[mpirank], local_rows[source]), dtype=np.float64)
            comm.Recv([local_recv_buf, MPI.DOUBLE], source=source, tag=mpirank)
            new_data[:, np.sum(local_rows[:source]):np.sum(local_rows[:source+1])] = local_recv_buf
    return new_data
    
def mpi_mult(A, B):
    """
    calculate matrix multiplication of two distributed data,
    the result is data1*data2 in multi-node distribution
    note that the numerical values will be converted into double
    
    we send the distributed B rows into other nodes (aka cannon method)
    
    parameters
    ----------
    
    A
        numpy.ndarray
        distributed left side data
        
    B
        numpy.ndarray
        distributed right side data
        
    return
    ------
    numpy.ndarray
    distributed multiplication result
    """
    assert (len(A.shape) == 2)
    assert (len(B.shape) == 2)
    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)
    # know the total rows
    C_global_row = np.array(0, dtype=np.uint)
    comm.Allreduce([np.array(A.shape[0]), MPI.LONG], [C_global_row, MPI.LONG], op=MPI.SUM)
    # prepare the distributed result
    C = np.zeros((A.shape[0], C_global_row), dtype=np.float64)
    # collect A and B matrix row info
    A_rows = np.empty(mpisize, dtype=np.uint)
    B_rows = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(A.shape[0], dtype=np.uint), MPI.LONG], [A_rows, MPI.LONG])
    comm.Allgather([np.array(B.shape[0], dtype=np.uint), MPI.LONG], [B_rows, MPI.LONG])
    assert (np.sum(B_rows) == A.shape[1])  # ensure A*B is legal
    # local mult with B cannons
    for itr in range(mpisize):
        # fire cannons
        target = (mpirank + itr) % mpisize
        source = (mpirank - itr) % mpisize
        local_sent_buf = np.array(B, dtype=np.float64)
        comm.Send([local_sent_buf, MPI.DOUBLE], dest=target, tag=target)
        local_recv_buf = np.empty((B_rows[source], B.shape[1]), dtype=np.float64)
        comm.Recv([local_recv_buf, MPI.DOUBLE], source=source, tag=mpirank)
        # accumulate local mult
        A_col_begin = np.sum(B_rows[:source])
        A_col_end = A_col_begin + B_rows[source]
        A_block = np.array(A[:, A_col_begin:A_col_end], dtype=np.float64)
        C = C + np.dot(A_block, local_recv_buf)
    return C