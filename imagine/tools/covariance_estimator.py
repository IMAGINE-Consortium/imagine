"""
This module contains estimation algorithms for the
covariance matrix based on a finite number of samples.

For the testing suits, please turn to "imagine/tests/tools_tests.py".
"""
import numpy as np
import logging as log
from imagine.tools.parallel_ops import pmean, ptrans, pmult, peye, ptrace, pshape

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
        Ensemble of observables, in global shape (ensemble size, data size).

    Returns
    -------
    cov : numpy.ndarray
        Distributed (not copied) covariance matrix in global shape (data size, data size),
        each node takes part of the rows.
    """
    log.debug('@ covariance_estimator::empirical_cov')
    assert isinstance(data, np.ndarray)
    assert (len(data.shape) == 2)
    # Get ensemble size (i.e. the number of rows)
    ensemble_size, _ = pshape(data)
    # Calculates covariance
    u = data - pmean(data)
    cov = pmult(ptrans(u), u) / ensemble_size
    return cov

def oas_cov(data):
    r"""
    Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    Given some :math:`n\times m` data matrix, :math:`D`,
    where rows are different samples and columns different properties,
    the covariance can be estimated in the following way.

    .. math::
           U_{ij} = D_{ij} -  \overline{D}_j\,,\;
           \text{with}\; \overline{D}_j=\tfrac{1}{n} \sum_{i=1}^n D_{ij}

    Let

    .. math::
          S = \tfrac{1}{n} U^T U\,,\;
          T = \text{tr}(S)\quad\text{and}\quad V = \text{tr}(S^2)

    .. math::
          \tilde\rho = \min\left[1,\frac{(1-2/m)V + T^2}{ (n+1-2/m)(V-T^2/m)}\right]

    The covariance is given by

    .. math::
          \text{cov}_\text{OAS} = (1-\rho)S + \tfrac{1}{N} \rho T I_m


    Parameters
    ----------
    data : numpy.ndarray
        Distributed data in global shape (ensemble_size, data_size).

    Returns
    -------
    cov : numpy.ndarray
        Covariance matrix in global shape (data_size, data_size).
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
        Distributed data in global shape (ensemble_size, data_size).

    Returns
    -------
    mean : numpy.ndarray
        Copied ensemble mean (on all nodes).
    cov : numpy.ndarray
        Distributed covariance matrix in shape (data_size, data_size).
    """
    log.debug('@ covariance_estimator::oas_mcov')
    assert isinstance(data, np.ndarray)
    assert (len(data.shape) == 2)

    # Finds ensemble size and data size
    data_size = data.shape[1]
    ensemble_size, _ = pshape(data)

    # Calculates OAS covariance extimator from empirical covariance estimator
    mean = pmean(data)
    u = data - mean
    s = pmult(ptrans(u), u) / ensemble_size
    trs = ptrace(s)
    trs2 = ptrace(pmult(s, s))

    numerator = (1.0 - 2.0/data_size)*trs2 + trs*trs
    denominator = (ensemble_size +1.0-2.0/data_size)*(trs2 - (trs*trs)/data_size)

    if denominator == 0:
        rho = 1
    else:
        rho = np.min([1, numerator/denominator])
    cov = (1.-rho)*s+peye(data_size)*rho*trs/data_size

    return mean, cov
