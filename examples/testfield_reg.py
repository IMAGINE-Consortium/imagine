"""
single TestField, regular field parameter constraints with mock data
error propagation is considered
which means measurements contain modelling but no observational uncertainty
the estimation uncertainty should be no larger than modelling uncertainty as verification benchmark
"""

import numpy as np
import logging as log

# visualize posterior
import corner
import json
import matplotlib
matplotlib.use('Agg')
from imagine.tools.carrier_mapper import unity_mapper

from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood
from imagine.likelihoods.simple_likelihood import SimpleLikelihood
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.priors.flat_prior import FlatPrior
from imagine.simulators.test.test_simulator import TestSimulator
from imagine.pipelines.pipeline import Pipeline

"""
OAS covariance estimator, sample comes with (Nens,Ndim) matrix
"""
def oas(_sample):
	[n, p] = np.shape(_sample)
	m = np.median(_sample, axis=0)
	u = _sample-m
	s = np.dot(u.T,u)/n
	trs = np.trace(s)
	trs2 = np.trace(np.dot(s,s))
	numerator = (1-2./p)*trs2+trs*trs
	denominator = (n+1.-2./p)*(trs2-(trs*trs)/p)
	if denominator == 0:
		rho = 1
	else:
		rho = np.min([1,numerator/denominator])
	return (1.-rho)*s+np.eye(p)*rho*trs/p

"""
activate parameters 'a' in TestField
"""
def testfield():

	log.basicConfig(filename='imagine.log', level=log.INFO)

	"""
	# step 0, set 'a', 'std_a'

	TestField is modeled as
		y = a*sin(x) + (gaussian_random(mean=0,std=b))**2
		where x in (0,2pi)

	for generating mock data we need
	true values of a
	theoretical uncertainties in a
	observational points, positioned in (0,2pi) evenly, due to TestField modelling
	parameter b is set by default, 0.
	"""
	true_a = 2.
	err_a = 0.1 # std of gaussian modelling error
	measure_points = 100  # data points in measurements
	measure_times = 100  # times of measures
	truths = [true_a]  # will be used in visualizing posterior

	"""
	# step 1, prepare mock data
	"""

	"""
	# 1.1, generate measurements and covariances
	"""
	x = np.linspace(0, 2. * np.pi, measure_points)
	measure_field = np.zeros((measure_times, measure_points))
	for i in range(measure_field.shape[0]):
		for j in range(measure_field.shape[1]):
			measure_field[i,j] = np.random.normal(true_a,err_a,1) * np.sin(x[j])
	measure_cov = oas(measure_field)  # get propagated covariance
	mock_data = Measurements()  # create empty Measrurements object
	mock_cov = Covariances()  # create empty Covariance object
	mock_data.append(('test', 'nan', str(measure_points), 'nan'),
					 np.vstack([measure_field[np.random.randint(measure_times)]]), True)
	mock_cov.append(('test', 'nan', str(measure_points), 'nan'), measure_cov, True)

	"""
	# 1.2, visualize mock data
	"""
	matplotlib.pyplot.plot(x, mock_data[('test', 'nan', str(measure_points), 'nan')].to_global_data()[0])
	matplotlib.pyplot.savefig('testfield_reg_mock.pdf')

	"""
	# step 2, prepare pipeline and execute analysis
	"""

	"""
	# 2.1, ensemble likelihood
	notice that with no random field, simulated ensemble has zero covariance
	"""
	# if discard mock_cov, ensemble likelihood acts like simple likelihood
	likelihood = EnsembleLikelihood(mock_data, mock_cov)  # initialize likelihood with measured info

	"""
	# 2.2, field factory list
	"""
	factory = TestFieldFactory(active_parameters=('a',))  # factory with single active parameter
	factory.parameter_ranges = {'a': (0, 10)}  # adjust parameter range for Bayesian analysis
	factory_list = [factory]  # likelihood requires a list/tuple of factories

	"""
	# 2.3, flat prior
	"""
	prior = FlatPrior()

	"""
	# 2.4, simulator 
	"""
	simer = TestSimulator(mock_data)

	"""
	# 2.5, pipeline
	"""
	ensemble_size = 10
	pipe = Pipeline(simer, factory_list, likelihood, prior, ensemble_size)
	pipe.random_seed = 0  # favor fixed seed? try a positive integer
	pipe()  # run with pymultinest

	"""
	# step 3, visualize (with corner package)
	"""
	with open('chains/imagine_params.json') as f:  # extract active parameter names
		params = json.load(f)
	points = np.loadtxt('chains/imagine_post_equal_weights.dat')  # load sample points
	for i in range(len(params)):  # convert variables into parameters
		low, high = pipe.active_ranges[params[i]]
		for j in range(points.shape[0]):
			points[j, i] = unity_mapper(points[j, i], low, high)
	figure = corner.corner(points[:, :len(params)],
						   range=[0.99] * len(params),
						   quantiles=[0.16, 0.5, 0.84], # 1 sigma -> [0.16, 0.5, 0.84]  2 sigma -> [0.02, 0.5, 0.98]
						   labels=params,
						   show_titles=True,
						   title_kwargs={"fontsize": 15},
						   color='steelblue',
						   truths=truths,
						   truth_color='k',
						   plot_contours=True,
						   hist_kwargs={'linewidth': 2},
						   label_kwargs={'fontsize': 15})
	matplotlib.pyplot.savefig('testfield_reg_posterior.pdf')

if __name__ == '__main__':
	testfield()