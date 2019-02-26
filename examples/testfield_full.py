"""
single TestField, full parameter constraints with mock data
observational/measreument covariance is pre-defined according to observational uncertainty
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
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.priors.flat_prior import FlatPrior
from imagine.simulators.test.test_simulator import TestSimulator
from imagine.pipelines.pipeline import Pipeline


def testfield():

	log.basicConfig(filename='imagine.log', level=log.INFO)

	"""
	# step 0,
	
	TestField is modeled as
		total_field = regular_field + variance_field
		y = a*sin(x) + (gaussian_random(mean=0,std=b))**2
		where x in (0,2pi)
	
	for generating mock data we need
	true values of a and b
	observational uncertainties in a and b
	observational points, positioned in (0,2pi) evenly, due to TestField modelling
	"""
	true_a = 6.
	true_b = 3.
	measure_err = 0.1 # std of gaussian measurement error
	measure_points = 100 # data points in measurements
	truths = [true_a, true_b]  # will be used in visualizing posterior

	"""
	# step 1, prepare mock data
	"""

	"""
	# 1.1, generate measurements and covariances
	total_field = regular_field + variance_field + noise_field
	"""
	x = np.linspace(0,2.*np.pi,measure_points)
	total_field = np.zeros((1,measure_points))
	regular_field = true_a * np.sin(x)
	np.random.seed(233)
	variance_field = (np.random.normal(0.,true_b,total_field.shape[1]))**2
	noise_field = np.random.normal(0.,measure_err,total_field.shape[1])

	total_field[0,:] += regular_field + variance_field + noise_field
	total_cov = (measure_err**2)*np.eye(total_field.shape[1])


	mock_data = Measurements() # create empty Measrurements object
	mock_cov = Covariances() # create empty Covariance object
	mock_data.append(('test', 'nan', str(measure_points), 'nan'), total_field, True)
	mock_cov.append(('test', 'nan', str(measure_points), 'nan'), total_cov, True)

	"""
	# 1.2, visualize mock data
	"""
	matplotlib.pyplot.plot(x, mock_data[('test', 'nan', str(measure_points), 'nan')].to_global_data()[0])
	matplotlib.pyplot.savefig('testfield_full_mock.pdf')

	"""
	# step 2, prepare pipeline and execute analysis
	"""

	"""
	# 2.1, ensemble likelihood
	"""
	likelihood = EnsembleLikelihood(mock_data, mock_cov) # initialize likelihood with measured info

	"""
	# 2.2, field factory list
	"""
	factory = TestFieldFactory(active_parameters=('a','b')) # factory with single active parameter
	factory.parameter_ranges = {'a':(0,10),'b':(0,10)} # adjust parameter range for Bayesian analysis
	factory_list = [factory] # likelihood requires a list/tuple of factories

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
	ensemble_size = 30
	pipe = Pipeline(simer, factory_list, likelihood, prior, ensemble_size)
	pipe.random_seed = 0 # favor fixed seed? try a positive integer
	pipe.pymultinest_parameter_dict = {'n_iter_before_update': 1, 'n_live_points': 400}
	result_dict = pipe() # run with pymultinest

	"""
	# step 3, visualize (with corner package)
	"""
	with open('chains/imagine_params.json') as f: # extract active parameter names
		params = json.load(f)
	points = np.loadtxt('chains/imagine_post_equal_weights.dat') # load sample points
	for i in range(len(params)): # convert variables into parameters
		low, high = pipe.active_ranges[params[i]]
		for j in range(points.shape[0]):
			points[j,i] = unity_mapper(points[j,i],low,high)
	figure = corner.corner(points[:, :len(params)],
						   range=[0.99]*len(params),
						   quantiles=[0.02, 0.5, 0.98], # 1 sigma -> [0.16, 0.5, 0.84]  2 sigma -> [0.02, 0.5, 0.98]
						   labels=params,
						   show_titles=True,
						   title_kwargs={"fontsize": 15},
						   color='steelblue',
						   truths=truths,
						   truth_color='k',
						   plot_contours=True,
						   hist_kwargs={'linewidth': 2},
						   label_kwargs={'fontsize': 15})
	matplotlib.pyplot.savefig('testfield_full_posterior.pdf')

if __name__ == '__main__':
	testfield()