"""
there are several ways to make robust estimation on covaraince matrix
based on finite number of samples
"""

import numpy as np

from imagine.observables.observable import Observable


def empirical_cov(_sample):
    """
    empirical covariance estimator
    the worst option
    :param _sample: ensemble of observables, in shape (ensemble_size,data_size)
    :return: covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(_sample, np.ndarray)
    n = _sample.shape[0]
    assert (n > 0)
    m = np.median(_sample, axis=0)
    u = _sample - m
    return np.dot(u.T, u) / n


def oas_cov(_sample):
    """
    OAS covariance estimator, prepared for examples
    :param _sample: ensemble of observables, in shape (ensemble_size,data_size)
    :return: covariance matrix in shape (data_size,data_size)
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


def bootstrap_cov(_sample, _trapsize=int(3000)):
    """
    bootstrap covariance estimator, prepared for examples
    :param _sample: ensemble of observables, in shape (ensemble_size,data_size)
    :param _trapsize: bootstrap volume
    :return: covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(_sample, np.ndarray)
    n, p = _sample.shape
    assert (n > 0 and p > 0)
    m = np.median(_sample, axis=0)
    u = _sample - m
    s_bst = np.zeros((p, p))
    for i in range(_trapsize):
        idx = np.random.randint(0, n, size=n)
        tmp = u[idx]
        s_bst = s_bst + np.dot(tmp.T, tmp)/n
    s_bst /= float(_trapsize)
    return s_bst


def oas_mcov(_sample):
    """
    OAS covariance estimator, prepared for Likelihood
    :param _sample: Observable object
    :return: ensemble mean, covariance matrix in shape (data_size,data_size)
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


def bootstrap_mcov(_sample, _trapsize=int(3000)):
    """
    estimate covariance with bootstrap
    :param _sample: ensemble of observables, in shape (ensemble_size,data_size)
    :param _trapsize: bootstrap volume
    :return: ensemble mean, covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(_sample, Observable)
    n, p = _sample.shape
    assert (n > 0 and p > 0)
    mean = _sample.ensemble_mean
    u = _sample.to_global_data() - mean
    s_bst = np.zeros((p, p))
    for i in range(_trapsize):
        idx = np.random.randint(0, n, size=n)
        tmp = u[idx]
        s_bst = s_bst + np.dot(tmp.T, tmp)/n
    s_bst /= float(_trapsize)
    return mean, s_bst


def trapoas_mcov(_sample, _trapsize=int(100)):
    """
    estimate covariance with bootstraped oas
    :param _sample: ensemble of observables, in shape (ensemble_size,data_size)
    :param _trapsize: bootstrap volume
    :return: ensemble mean, covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(_sample, Observable)
    n, p = _sample.shape
    assert (n > 0 and p > 0)
    mean = _sample.ensemble_mean
    u = _sample.to_global_data() - mean
    s_bst = np.zeros((p, p))
    for i in range(_trapsize):
        idx = np.random.randint(0, n, size=n)
        tmp = u[idx]
        s_bst = s_bst + oas_cov(tmp)
    s_bst /= float(_trapsize)
    return mean, s_bst
