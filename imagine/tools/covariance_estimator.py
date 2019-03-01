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
	:param _sample: Observable object
	:return: ensemble median, covariance matrix in shape (data_size,data_size)
	"""
	assert isinstance(_sample, Observable)
	n, p = _sample.shape
	assert (p > 0 and n > 1)
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


def bootstrap_cov(_sample):
	"""
	estimate covariance with bootstrap
	convergence may appear after 3k iterations
	:param _sample: ensemble of observables, in shape (ensemble_size,data_size)
	:return: covariance matrix in shape (data_size,data_size)
	"""
	pass
