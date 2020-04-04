"""
This module targets on estimation algorithms for covaraince matrix
based on finite number of samples.
For the testing suits, please turn to "imagine/tests/tools_tests.py".
"""

import numpy as np
from mpi4py import MPI
import logging as log
from imagine.tools.mpi_helper import mpi_mean, mpi_trans, mpi_mult, mpi_eye, mpi_trace


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

def empirical_cov(data):
    """
    empirical covariance estimator
    for distributed data with multiple global rows
    (probably) the worst option
    
    Parameters
    ----------
    
    sample : distributed numpy.ndarray
        ensemble of observables, in global shape (ensemble size, data size)
        
    Returns
    -------
    numpy.ndarray
    distributed (not copied) covariance matrix in global shape (data size, data size)
    each node takes part of the rows
    """
    log.debug('@ covariance_estimator::empirical_cov')
    assert isinstance(data, np.ndarray)
    assert (len(data.shape) == 2)
    # Get ensemble size (i.e. the number of rows)
    ensemble_size = np.array(0, dtype=np.uint64)
    comm.Allreduce([np.array(data.shape[0], dtype=np.uint64),
                    MPI.LONG], [ensemble_size, MPI.LONG],
                   op=MPI.SUM)
    # Calculates covariance
    u = data - mpi_mean(data)
    cov = mpi_mult(mpi_trans(u), u) / ensemble_size
    return cov

def oas_cov(data):
    """
    OAS covariance estimator, prepared for examples
    
    Paramters
    ---------
    
    data : numpy.ndarray
        distributed data in global shape (ensemble_size, data_size)
        
    Returns
    -------
    distributed numpy.ndarray
    covariance matrix in global shape (data_size, data_size)
    """
    log.debug('@ covariance_estimator::oas_cov')
    assert isinstance(data, np.ndarray)
    assert (len(data.shape) == 2)
    # data size
    data_size = data.shape[1]
    # ensemble size
    ensemble_size = np.array(0, dtype=np.uint)
    comm.Allreduce([np.array(data.shape[0], dtype=np.uint), MPI.LONG], [ensemble_size, MPI.LONG], op=MPI.SUM)
    # calculate OAS covariance extimator from empirical covariance estimator
    u = data - mpi_mean(data)
    s = mpi_mult(mpi_trans(u), u) / ensemble_size
    trs = mpi_trace(s)
    trs2 = mpi_trace(mpi_mult(s, s))
    numerator = (1.0-2.0/data_size)*trs2+trs*trs
    denominator = (ensemble_size+1.0-2.0/data_size)*(trs2-(trs*trs)/data_size)
    if denominator == 0:
        rho = 1
    else:
        rho = np.min([1, numerator/denominator])
    return (1.-rho)*s+mpi_eye(data_size)*rho*trs/data_size

def oas_mcov(data):
    """
    OAS covariance estimator, prepared for Likelihood
    
    Parameters
    ----------
    
    data : numpy.ndarray
        distributed data in global shape (ensemble_size, data_size)
        
    Returns
    -------
    it returns two results
    copied ensemble mean (on all nodes)
    distributed covariance matrix in shape (data_size, data_size)
    """
    log.debug('@ covariance_estimator::oas_mcov')
    assert isinstance(data, np.ndarray)
    assert (len(data.shape) == 2)
    # data size
    data_size = data.shape[1]
    ensemble_size = np.array(0, dtype=np.uint64)
    comm.Allreduce([np.array(data.shape[0], dtype=np.uint64), MPI.LONG],
                   [ensemble_size, MPI.LONG], op=MPI.SUM)

    # Calculates OAS covariance extimator from empirical covariance estimator
    mean = mpi_mean(data)
    u = data - mean
    s = mpi_mult(mpi_trans(u), u) / ensemble_size
    trs = mpi_trace(s)
    trs2 = mpi_trace(mpi_mult(s, s))
    numerator = (1.0-2.0/data_size)*trs2+trs*trs
    denominator = (ensemble_size+1.0-2.0/data_size)*(trs2-(trs*trs)/data_size)
    if denominator == 0:
        rho = 1
    else:
        rho = np.min([1, numerator/denominator])
    return mean, (1.-rho)*s+mpi_eye(data_size)*rho*trs/data_size
