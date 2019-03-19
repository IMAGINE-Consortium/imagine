"""
single TestField with LiSimulator
full parameter constraints with mock data
"""

import numpy as np
import logging as log

from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood
from imagine.likelihoods.simple_likelihood import SimpleLikelihood
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.priors.flat_prior import FlatPrior
from imagine.simulators.test.li_simulator import LiSimulator
from imagine.pipelines.dynesty_pipeline import DynestyPipeline
from imagine.tools.covariance_estimator import oas_cov

# visualize posterior
import corner
import matplotlib
from imagine.tools.carrier_mapper import unity_mapper
matplotlib.use('Agg')


def testfield():
    """

    :return:

    log.basicConfig(filename='imagine.log', level=log.INFO)
    """

    """
    # step 0, set 'a' and 'b', 'mea_std'
    
    TestField in LiSimulator is modeled as
        field = gaussian_random(mean=a,std=b)_x * cos(x)
        where x in (0,2pi)
    
    for generating mock data we need
    true values of a and b: true_a, true_b, mea_seed
    measurement uncertainty: mea_std
    measurement points, positioned in (0,2pi) evenly, due to TestField modelling
    """
    true_a = 3.
    true_b = 6.
    mea_std = 0.1  # std of gaussian measurement error
    mea_seed = 233
    mea_points = 10  # data points in measurements
    truths = [true_a, true_b]  # will be used in visualizing posterior

    """
    # step 1, prepare mock data
    """

    """
    # 1.1, generate measurements
    mea_field = signal_field + noise_field
    """
    x = np.linspace(0, 2.*np.pi, mea_points)
    np.random.seed(mea_seed)  # seed for signal field
    signal_field = np.multiply(np.cos(x),
                               np.random.normal(loc=true_a, scale=true_b, size=mea_points))
    mea_field = np.vstack([signal_field + np.random.normal(loc=0., scale=mea_std, size=mea_points)])

    """
    # 1.2, generate covariances
    what's the difference between pre-define dan re-estimated?
    """
    # re-estimate according to measurement error
    repeat = 100  # times of repeated measurements
    mea_repeat = np.zeros((repeat, mea_points))
    for i in range(repeat):
        mea_repeat[i, :] = signal_field + np.random.normal(loc=0., scale=mea_std, size=mea_points)
    mea_cov = oas_cov(mea_repeat)

    print('re-estimated: \n', mea_cov)

    # pre-defined according to measurement error
    mea_cov = (mea_std**2) * np.eye(mea_points)

    print('pre-defined: \n', mea_cov)

    """
    # 1.3 assemble in imagine convention
    """

    mock_data = Measurements()  # create empty Measrurements object
    mock_cov = Covariances()  # create empty Covariance object
    # pick up a measurement
    mock_data.append(('test', 'nan', str(mea_points), 'nan'), mea_field, True)
    mock_cov.append(('test', 'nan', str(mea_points), 'nan'), mea_cov, True)

    """
    # 1.4, visualize mock data
    """
    matplotlib.pyplot.plot(x, mock_data[('test', 'nan', str(mea_points), 'nan')].to_global_data()[0])
    matplotlib.pyplot.savefig('testfield_mock.pdf')

    """
    # step 2, prepare pipeline and execute analysis
    """

    """
    # 2.1, ensemble likelihood
    """
    likelihood = EnsembleLikelihood(mock_data, mock_cov)  # initialize likelihood with measured info
    #likelihood = SimpleLikelihood(mock_data, mock_cov)
    #likelihood.active_parameters = ()

    """
    # 2.2, field factory list
    """
    factory = TestFieldFactory(active_parameters=('a', 'b'))  # factory with single active parameter
    factory.parameter_ranges = {'a': (0, 10), 'b': (0, 10)}  # adjust parameter range for Bayesian analysis
    factory_list = [factory]  # likelihood requires a list/tuple of factories

    """
    # 2.3, flat prior
    """
    prior = FlatPrior()

    """
    # 2.4, simulator 
    """
    simer = LiSimulator(mock_data)

    """
    # 2.5, pipeline
    """
    ensemble_size = 10
    pipe = DynestyPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_type = 'controlable'  # 'fixed' wont work for Dynesty
    pipe.seed_tracer = int(23)
    pipe.sampling_controllers = {'nlive': 400}
    results = pipe()  # run with pymultinest

    """
    # step 3, visualize (with corner package)
    """
    samples = results['samples']
    for i in range(len(pipe.active_parameters)):  # convert variables into parameters
        low, high = pipe.active_ranges[pipe.active_parameters[i]]
        for j in range(samples.shape[0]):
            samples[j, i] = unity_mapper(samples[j, i], low, high)
    # corner plot
    corner.corner(samples[:, :len(pipe.active_parameters)],
                  range=[0.99] * len(pipe.active_parameters),
                  quantiles=[0.02, 0.5, 0.98],
                  labels=pipe.active_parameters,
                  show_titles=True,
                  title_kwargs={"fontsize": 15},
                  color='steelblue',
                  truths=truths,
                  truth_color='firebrick',
                  plot_contours=True,
                  hist_kwargs={'linewidth': 2},
                  label_kwargs={'fontsize': 20})
    matplotlib.pyplot.savefig('testfield_posterior.pdf')


if __name__ == '__main__':
    testfield()
