"""
This module contains estimation algorithms for the
covariance matrix based on a finite number of samples.

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
    r"""
    Empirical covariance estimator


    Given some data matrix, :math:`D`, where rows are different samples
    and columns different properties, the covariance can be
    estimated from

    .. math::
           U_{ij} = D_{ij} -  \overline{D}_j\,,\;
           \text{with}\; \overline{D}_j=\tfrac{1}{N} \sum_{i=1}^N D_{ij}

    .. math::
          \text{cov} = \tfrac{1}{N} U^T U



    Notes
    -----
        While conceptually simple, this is usually not the
        best option.

    Parameters
    ----------
    data : numpy.ndarray
        ensemble of observables, in global shape (ensemble size, data size)

    Returns
    -------
    cov : numpy.ndarray
        distributed (not copied) covariance matrix in global shape
        (data size, data size) each node takes part of the rows
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
    r"""
    Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    Given some :math:`n\times m` data matrix, :math:`D`,
    where rows are different samples and columns different properties,
    the covariance can be estimated in the following way.

    .. math::
           U_{ij} = D_{ij} -  \overline{D}_j\,,\;
           \text{with}\; \overline{D}_j=\tfrac{1}{N} \sum_{i=1}^N D_{ij}

    Let

    .. math::
          S = \tfrac{1}{N} U^T U\,,\;
          t = \text{tr}(S)\quad\text{and}\quad v = \text{tr}(S)

    .. math::
          \tilde\rho = \min\left[1,\frac{(1-2/m)v + t^2}{ (m+1-2/m)(v-t^2/m)}\right]

    The covariance is given by

    .. math::
          \text{cov}_\text{OAS} = (1-\rho)S + \tfrac{1}{N} t \rho I_m


    Parameters
    ----------
    data : numpy.ndarray
        distributed data in global shape (ensemble_size, data_size)

    Returns
    -------
    cov : numpy.ndarray
        covariance matrix in global shape (data_size, data_size)
    """
    log.debug('@ covariance_estimator::oas_cov')
    _, cov = oas_mcov(data)

    return cov

def oas_mcov(data):
    """
    Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    See `imagine.tools.covariance_estimator.oas_cov` for details. This
    function aditionally returns the computed ensemble mean.

    Parameters
    ----------
    data : numpy.ndarray
        distributed data in global shape (ensemble_size, data_size)

    Returns
    -------
    mean : numpy.ndarray
        copied ensemble mean (on all nodes)
    cov : numpy.ndarray
        distributed covariance matrix in shape (data_size, data_size)
    """
    log.debug('@ covariance_estimator::oas_mcov')
    assert isinstance(data, np.ndarray)
    assert (len(data.shape) == 2)

    # Finds ensemble size and data size
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

    numerator = (1.0 - 2.0/data_size)*trs2 + trs*trs
    denominator = (ensemble_size +1.0-2.0/data_size)*(trs2 - (trs*trs)/data_size)

    if denominator == 0:
        rho = 1
    else:
        rho = np.min([1, numerator/denominator])
    cov = (1.-rho)*s+mpi_eye(data_size)*rho*trs/data_size

    return mean, cov
