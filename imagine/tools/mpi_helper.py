"""
This MPI helper module is designed for parallel computing and data handling.

For the testing suits, please turn to "imagine/tests/tools_tests.py".
"""

import numpy as np
from mpi4py import MPI
from copy import deepcopy
import logging as log


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

def mpi_arrange(size):
    """
    With known global size, number of mpi nodes, and current rank,
    returns the begin and end index for distributing the global size.
    
    Parameters
    ----------
    
    size : integer (positive)
        The total size of target to be distributed.
        It can be a row size or a column size.
    
    Returns
    -------
    result : numpy.uint
        The begin and end index [begin,end] for slicing the target.
    """
    log.debug('@ mpi_helper::mpi_arrange')
    assert (size > 0)
    res = min(mpirank, size%mpisize)
    ave = size//mpisize
    if (ave == 0):
        raise ValueError('over distribution')
    return np.uint(res + mpirank*ave), np.uint(res + (mpirank+1)*ave + 
                                               np.uint(mpirank < size%mpisize))

def mpi_shape(data):
    """
    Returns the global number of rows and columns of given distributed data.
    
    Parameters
    ----------
    
    data : numpy.ndarray
        The distributed data.
        
    Returns
    -------
    result : numpy.uint
        Glboal row and column number.
    """
    global_row = np.array(0, dtype=np.uint)
    comm.Allreduce([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [global_row, MPI.LONG], op=MPI.SUM)
    global_column = np.array(data.shape[1], dtype=np.uint)
    return global_row, global_column

def mpi_prosecutor(data):
    """
    Check if the data is distributed in the correct way
    covariance matrix is distributed exactly the same manner as multi-realization data
    if not, an error will be raised.
    
    Parameters
    ----------
    
    data : numpy.ndarray
        The distributed data to be examined.
    """
    log.debug('@ mpi_helper::mpi_prosecutor')
    assert isinstance(data, np.ndarray)
    # get the global shape
    local_rows = np.empty(mpisize, dtype=np.uint)
    local_cols = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
    comm.Allgather([np.array(data.shape[1], dtype=np.uint), MPI.LONG], [local_cols, MPI.LONG])
    check_begin, check_end = mpi_arrange(np.sum(local_rows))
    if (data.shape[0] != check_end - check_begin):
        raise ValueError('incorrect data allocation')
    if np.any((local_cols-local_cols[0]).astype(bool)):
        raise ValueError('incorrect data allocation')
        
def mpi_mean(data):
    """
    calculate the mean of distributed array
    prefers averaging along column direction
    but if given (1,n) data shape
    the average is done along row direction the result
    note that the numerical values will be converted into double
    
    Parameters
    ----------
    
    data : numpy.ndarray
        Distributed data.
        
    Returns
    -------
    result : numpy.ndarray
        Copied data mean, which means the mean is copied to all nodes.
    """
    log.debug('@ mpi_helper::mpi_mean')
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
    Transpose distributed data,
    note that the numerical values will be converted into double.
    
    Parameters
    ----------
    
    data : numpy.ndarray
        Distributed data.
        
    Returns
    -------
    result : numpy.ndarray
        Transposed data in distribution.
    """
    log.debug('@ mpi_helper::mpi_trans')
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
            comm.Isend([local_sent_buf, MPI.DOUBLE], dest=target, tag=target)
        else:  # recv from self
            # note that transpose works here (in-node copy)
            new_col_begin = np.sum(local_rows[0:mpirank])
            new_col_end = new_col_begin + local_rows[mpirank]
            new_data[:, new_col_begin:new_col_end] = np.transpose((data[:, cut_col_begin:cut_col_end]).astype(np.float64))
    for source in range(mpisize):
        if (source != mpirank):  # recv from other ranks
            local_recv_buf = np.empty((cut_col_ends[mpirank]-cut_col_begins[mpirank], local_rows[source]), dtype=np.float64)
            comm.Recv([local_recv_buf, MPI.DOUBLE], source=source, tag=mpirank)
            new_data[:, np.sum(local_rows[0:source]):np.sum(local_rows[0:source+1])] = local_recv_buf
    return new_data
    
def mpi_mult(left, right):
    """
    Calculate matrix multiplication of two distributed data,
    the result is data1*data2 in multi-node distribution
    note that the numerical values will be converted into double.
    We send the distributed right rows into other nodes (aka cannon method).
    
    Parameters
    ----------
    
    left : numpy.ndarray
        Distributed left side data.
        
    right : numpy.ndarray
        Distributed right side data.
        
    Returns
    -------
    result : numpy.ndarray
        Distributed multiplication result.
    """
    log.debug('@ mpi_helper::mpi_mult')
    assert (len(left.shape) == 2)
    assert (len(right.shape) == 2)
    assert isinstance(left, np.ndarray)
    assert isinstance(right, np.ndarray)
    # know the total rows
    result_global_row = np.array(0, dtype=np.uint)
    comm.Allreduce([np.array(left.shape[0], dtype=np.uint), MPI.LONG], [result_global_row, MPI.LONG], op=MPI.SUM)
    # prepare the distributed result
    result = np.zeros((left.shape[0], result_global_row), dtype=np.float64)
    # collect left and right matrix row info
    left_rows = np.empty(mpisize, dtype=np.uint)
    right_rows = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(left.shape[0], dtype=np.uint), MPI.LONG], [left_rows, MPI.LONG])
    comm.Allgather([np.array(right.shape[0], dtype=np.uint), MPI.LONG], [right_rows, MPI.LONG])
    assert (np.sum(right_rows) == left.shape[1])  # ensure left*right is legal
    # local self mult
    left_col_begin = np.sum(right_rows[:mpirank])
    left_col_end = left_col_begin + right_rows[mpirank]
    left_block = np.array(left[:, left_col_begin:left_col_end], dtype=np.float64)
    result += np.dot(left_block, right)
    # local mult with right cannons
    # allocate fixed bufs
    for itr in range(1, mpisize):
        target = (mpirank + itr) % mpisize
        source = (mpirank - itr) % mpisize
        # fire cannons
        comm.Isend([right.astype(np.float64), MPI.DOUBLE], dest=target, tag=target)
        # receive cannons
        local_recv_buf = np.zeros((right_rows[source], right.shape[1]), dtype=np.float64)
        comm.Recv([local_recv_buf, MPI.DOUBLE], source=source, tag=mpirank)
        # accumulate local mult
        left_col_begin = np.sum(right_rows[0:source])
        left_col_end = left_col_begin + right_rows[source]
        left_block = np.array(left[:, left_col_begin:left_col_end], dtype=np.float64)
        result += np.dot(left_block, local_recv_buf)
    return result

def mpi_trace(data):
    """
    Computes the trace of the given distributed data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Array of data distributed over different processes.
        
    Returns
    -------
    result : numpy.float64
        Copied trace of given data.
    """
    log.debug('@ mpi_helper::mpi_trace')
    assert (len(data.shape) == 2)
    assert isinstance(data, np.ndarray)
    local_acc = np.array(0, dtype=np.float64)
    local_row_begin, local_row_end = mpi_arrange(data.shape[1])
    for i in range(local_row_end - local_row_begin):
        eye_pos = local_row_begin + np.uint(i)
        local_acc += np.float64(data[i, eye_pos])
    result = np.array(0, dtype=np.float64)
    comm.Allreduce([local_acc, MPI.DOUBLE], [result, MPI.DOUBLE], op=MPI.SUM)
    return result
    
def mpi_eye(size):
    """
    Produces an eye matrix according of shape (size,size)
    distributed over the various running MPI processes
    
    Parameters
    ----------
    size : integer
        Distributed matrix size.
        
    Returns
    -------
    result : numpy.ndarray, double data type
        Distributed eye matrix.
    """
    log.debug('@ mpi_helper::mpi_eye')
    local_row_begin, local_row_end = mpi_arrange(size)
    local_matrix = np.zeros((local_row_end - local_row_begin, size), dtype=np.float64)
    for i in range(local_row_end - local_row_begin):
        eye_pos = local_row_begin + np.uint(i)
        local_matrix[i, eye_pos] =  1.0
    return local_matrix

def mpi_lu_solve(operator, source):
    """
    Simple LU Gauss method WITHOUT pivot permutation.
    
    Parameters
    ----------
    operator : distributed numpy.ndarray
        Matrix representation of the left-hand-side operator.
        
    source : copied numpy.ndarray
        Vector representation of the right-hand-side source.
        
    Returns
    -------
    result : numpy.ndarray, double data type
        Copied solution to the linear algebra problem.
    """
    log.debug('@ mpi_helper::mpi_lu_solve')
    assert isinstance(operator, np.ndarray)
    assert isinstance(source, np.ndarray)
    global_rows = operator.shape[1]
    assert (source.shape == (1, global_rows))
    u = deepcopy(operator.astype(np.float64))
    x = deepcopy(source.astype(np.float64))
    # split x
    xsplit_begin, xsplit_end = mpi_arrange(global_rows)
    xsplit = np.array(x[0,xsplit_begin:xsplit_end])
    # collect local rows for each node
    local_rows = np.empty(mpisize, dtype=np.uint)
    xsplit_begins = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(u.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
    comm.Allgather([np.array(xsplit_begin, dtype=np.uint), MPI.LONG], [xsplit_begins, MPI.LONG])
    # start gauss method
    # goes column by column
    for c in range(global_rows-1):
        # find the pivot rank and local row
        pivot_rank = np.array(0, dtype=np.uint)
        pivot_r = np.uint(c)  # local row index hosting the pivot
        for i in range(len(local_rows)):
            if (pivot_r >= local_rows[i]):
                pivot_r -= local_rows[i]
            else:
                pivot_rank = np.uint(i)
                break
        # propagate pivot rank and pivot row
        if mpirank == pivot_rank:
            pivot_row = np.array(u[pivot_r, :])
        else:
            pivot_row = np.empty(global_rows, dtype=np.float64)
        comm.Bcast([np.array(pivot_rank), MPI.LONG], root=pivot_rank)
        comm.Bcast([pivot_row, MPI.DOUBLE], root=pivot_rank)
        # gauss elimination
        max_row = max(local_rows)
        for local_r in range(max_row):
            if (local_r + xsplit_begin > c and local_r < local_rows[mpirank]):
                ratio = u[local_r, c]/pivot_row[c]
                u[local_r,:] -= ratio*pivot_row
                xsplit[local_r] -= ratio*x[0, c]  # manipulate split x instead x
            # gather xsplit
            comm.Allgatherv([xsplit, MPI.DOUBLE], [x, local_rows, xsplit_begins, MPI.DOUBLE])
    # solve Ux=b
    for i in range(mpisize):
        op_rank = mpisize - 1 - i  # operational rank
        if (mpirank == op_rank):
            for j in range(local_rows[mpirank]):
                local_r = np.uint(local_rows[mpirank] - 1 - j)
                local_c = np.uint(xsplit_begin + local_r)
                local_c_plus = np.uint(local_c + 1)
                x[0,local_c] = (x[0,local_c] -
                                np.dot(u[local_r, local_c_plus:],
                                       x[0,local_c_plus:]) 
                               )/u[local_r,local_c]
        # update x
        comm.Bcast([x, MPI.DOUBLE], root=op_rank)
    return x
                
def mpi_slogdet(data):
    """
    Computes log determinant according to
    simple LU Gauss method WITHOUT pivot permutation.
        
    Parameters
    ----------
    data : numpy.ndarray
        Array of data distributed over different processes.
        
    Returns
    -------
    sign : numpy.ndarray
        Single element numpy array containing the sign of the determinant (copied to all nodes).
    logdet : numpy.ndarray
        Single element numpy array containing the log of the determinant (copied to all nodes).
    """
    log.debug('@ mpi_helper::mpi_slogdet')
    assert isinstance(data, np.ndarray)
    global_rows = data.shape[1]
    u = deepcopy(data.astype(np.float64))
    # collect local rows for each node
    local_rows = np.empty(mpisize, dtype=np.uint)
    comm.Allgather([np.array(u.shape[0], dtype=np.uint), MPI.LONG], [local_rows, MPI.LONG])
    # start gauss method
    # the hidden global row count in other nodes
    global_row_begin = np.sum(local_rows[0:mpirank])
    # goes column by column
    for c in range(global_rows-1):
        # find the pivot rank and local row
        pivot_rank = np.array(0, dtype=np.uint)
        pivot_r = np.uint(c)  # local row index hosting the pivot
        for i in range(len(local_rows)):
            if (pivot_r >= local_rows[i]):
                pivot_r -= local_rows[i]
            else:
                pivot_rank = np.uint(i)
                break
        # propagate pivot rank and pivot row
        if mpirank == pivot_rank:
            pivot_row = np.array(u[pivot_r, :])
        else:
            pivot_row = np.empty(global_rows, dtype=np.float64)
        comm.Bcast([np.array(pivot_rank), MPI.LONG], root=pivot_rank)
        comm.Bcast([pivot_row, MPI.DOUBLE], root=pivot_rank)
        # gauss elimination
        max_row = max(local_rows)
        for local_r in range(max_row):
            if (local_r + global_row_begin > c and local_r < local_rows[mpirank]):
                ratio = u[local_r, c]/pivot_row[c]
                u[local_r,:] -= ratio*pivot_row
    # calculate diagonal mult in the upper matrix
    sign = np.array(1.0, dtype=np.float64)
    logdet = np.array(0.0, dtype=np.float64)
    local_sign = np.array(1.0, dtype=np.float64)
    local_logdet = np.array(0.0, dtype=np.float64)
    for local_r in range(local_rows[mpirank]):
        local_c = np.uint(local_r + global_row_begin)
        target = u[local_r, local_c]
        target_sign = 2.0*np.float64(target>0) - 1.0
        local_sign *= target_sign
        local_logdet += np.log(target*target_sign)
    # reduce local diagonal element mult
    comm.Allreduce([local_logdet, MPI.DOUBLE], [logdet, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce([local_sign, MPI.DOUBLE], [sign, MPI.DOUBLE], op=MPI.PROD)
    assert (logdet != 0 and sign != 0)
    return sign, logdet

def mpi_global(data):
    """
    Gathers data spread accross different processes.
        
    Parameters
    ----------
    data : numpy.ndarray
        Array of data distributed over different processes.
        
    Returns
    -------
    global array : numpy.ndarray
        The root process returns the gathered data,
        other processes return `None`.
    """
    local_rows = np.array(data.shape[0], dtype=np.uint)
    global_rows = np.array(0, dtype=np.uint)
    comm.Allreduce([local_rows, MPI.LONG], [global_rows, MPI.LONG], op=MPI.SUM)
    local_row_begin, local_row_end = mpi_arrange(global_rows)
    if not mpirank:
        global_array = np.empty((global_rows, data.shape[1]), dtype=np.float64)
        row_begins = np.empty(mpisize, dtype=np.uint)
        row_ends = np.empty(mpisize, dtype=np.uint)
    else:
        global_array = None
        row_begins = None
        row_ends = None
    comm.Gather([np.array(local_row_begin, dtype=np.uint), MPI.LONG], [row_begins, MPI.LONG], root=0)
    comm.Gather([np.array(local_row_end, dtype=np.uint), MPI.LONG], [row_ends, MPI.LONG], root=0)
    if not mpirank:
        global_array[:local_row_end,:] = data
        for source in range(1,mpisize):
            comm.Recv([global_array[row_begins[source]:row_ends[source],:], MPI.DOUBLE] ,source=source, tag=source)
    else:
        pass
        comm.Isend([np.array(data, dtype=np.float64), MPI.DOUBLE], dest=0, tag=mpirank)
    return global_array
    """
    # the alternative way
    global_array = comm.gather(data, root=root)
    if global_array is not None:
        global_array = np.vstack(global_array)
    return global_array
    """

def mpi_local(data):
    """
    Distributes data over available processes
    
    Parameters
    ----------
    data : numpy.ndarray
        Array of data to be distributed over available processes.
        
    Returns
    -------
    local array : numpy.ndarray
        Return the distributed array on all preocesses.
    """
    if not mpirank:
        global_shape = np.array(data.shape, dtype=np.uint)
        row_begins = np.empty(mpisize, dtype=np.uint)
        row_ends = np.empty(mpisize, dtype=np.uint)
    else:
        global_shape = np.empty(2, dtype=np.uint)
        row_begins = None
        row_ends = None
    comm.Bcast([global_shape, MPI.LONG], root=0)
    local_row_begin, local_row_end = mpi_arrange(global_shape[0])
    comm.Gather([np.array(local_row_begin, dtype=np.uint), MPI.LONG], [row_begins, MPI.LONG], root=0)
    comm.Gather([np.array(local_row_end, dtype=np.uint), MPI.LONG], [row_ends, MPI.LONG], root=0)
    # start slicing
    if not mpirank:
        local_data = np.array(data[local_row_begin:local_row_end,:], dtype=np.float64)
        for target in range(1,mpisize):
            sendbuf = np.array(data[row_begins[target]:row_ends[target],:], dtype=np.float64)
            comm.Send([sendbuf, MPI.DOUBLE], dest=target, tag=target)
    else:
        local_data = np.empty((local_row_end-local_row_begin,global_shape[1]), dtype=np.float64)
        comm.Recv([local_data, MPI.DOUBLE], source=0, tag=mpirank)
    return local_data
