"""
There are several ways to make robust estimation on covaraince matrix
based on finite number of samples.
For the testing suits, please turn to "imagine/tests/tools_tests.py".
"""

import numpy as np
from mpi4py import MPI
from imagine.tools.mpi_helper import mpi_mean
#from imagine.observables.observable import Observable

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

'''
def empirical_cov(data):
    """
    empirical covariance estimator
    for distributed data with multiple global rows
    (probably) the worst option
    
    parameters
    ----------
    
    sample
        distributed numpy.ndarray
        ensemble of observables, in global shape (ensemble size, data size)
        
    return
    ------
    numpy.ndarray
    distributed (not copied) covariance matrix in global shape (data size, data size)
    each node takes part of the rows
    """
    assert isinstance(data, np.ndarray)
    assert (len(data.shape) == 2)
    # get ensemble size
    u = data - mpi_mean(data)  # copied mean
    return np.dot(u.T, u) / n
'''

'''
def oas_cov(_sample):
    """
    OAS covariance estimator, prepared for examples
    
    paramters
    ---------
    
    _sample
        numpy.ndarray
        ensemble of observables, in shape (ensemble_size,data_size)
        
    return
    ------
    
    covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(_sample, np.ndarray)
    n, p = _sample.shape
    assert (n > 0 and p > 0)
    if n == 1:
        return np.zeros((p, p))
    m = np.median(_sample, axis=0)
    u = _sample-m
    s = np.dot(u.T, u)/n
    trs = np.trace(s)
    trs2 = np.trace(np.dot(s, s))
    numerator = (1-2./p)*trs2+trs*trs
    denominator = (n+1.-2./p)*(trs2-(trs*trs)/p)
    if denominator == 0:
        rho = 1
    else:
        rho = np.min([1, numerator/denominator])
    return (1.-rho)*s+np.eye(p)*rho*trs/p

def oas_mcov(_sample):
    """
    OAS covariance estimator, prepared for Likelihood
    
    parameters
    ----------
    
    _sample
        Observable object
        
    return
    ------
    
    ensemble mean, covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(_sample, Observable)
    n, p = _sample.shape
    assert (n > 0 and p > 0)
    if n == 1:
        return _sample.to_global_data(), np.zeros((p, p))
    mean = _sample.ensemble_mean
    u = _sample.to_global_data() - mean
    # empirical covariance S
    s = np.dot(u.T, u)/n
    assert (s.shape[0] == u.shape[1])
    # Tr(S), equivalent to np.vdot(u,u)/n
    trs = np.trace(s)
    # Tr(S^2), equivalent to (np.einsum(u,[0,1],u,[2,1])**2).sum()/(n**2)
    trs2 = np.trace(np.dot(s, s))
    numerator = (1.-2./p)*trs2 + trs*trs
    denominator = (n+1.-2./p) * (trs2-(trs*trs)/p)
    if denominator == 0:
        rho = 1
    else:
        rho = np.min([1., float(numerator/denominator)])
    return mean, (1.-rho)*s + np.eye(p)*rho*trs/p
'''