"""
single TestField with LiSimulator
full parameter constraints with mock data
(with MPI support)
"""

import numpy as np
import logging as log
from mpi4py import MPI
from imagine.observables.observable_dict import Measurements, Covariances
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood
from imagine.fields.test_field.test_field_factory import TestFieldFactory
from imagine.priors.flat_prior import FlatPrior
from imagine.simulators.test.li_simulator import LiSimulator
from imagine.pipelines.multinest_pipeline import MultinestPipeline
from imagine.tools.mpi_helper import mpi_eye
from imagine.tools.timer import Timer


comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()
# visualize posterior
import corner, matplotlib
import matplotlib.pyplot as plt
from imagine.tools.carrier_mapper import unity_mapper
matplotlib.use('Agg')


def testfield(measure_size, simulation_size, debug=False):
    if debug:
        log.basicConfig(filename='imagine_li_multinest.log', level=log.DEBUG)
    else:
        log.basicConfig(filename='imagine_li_multinest.log')
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
    truths = [true_a, true_b]  # will be used in visualizing posterior
    """
    # step 1, prepare mock data
    """
    """
    # 1.1, generate measurements
    mea_field = signal_field + noise_field
    """
    x = np.linspace(0, 2.*np.pi, measure_size)  # data points in measurements
    np.random.seed(mea_seed)  # seed for signal field
    signal_field = np.multiply(np.cos(x),
                               np.random.normal(loc=true_a, scale=true_b, size=measure_size))
    mea_field = np.vstack([signal_field + np.random.normal(loc=0., scale=mea_std, size=measure_size)])

    """
    # 1.2, generate covariances
    """
    # pre-defined according to measurement error
    mea_cov = (mea_std**2) * mpi_eye(measure_size)

    """
    # 1.3 assemble in imagine convention
    """

    mock_data = Measurements()  # create empty Measrurements object
    mock_cov = Covariances()  # create empty Covariance object
    # pick up a measurement
    mock_data.append(('test', 'nan', str(measure_size), 'nan'), mea_field, True)
    mock_cov.append(('test', 'nan', str(measure_size), 'nan'), mea_cov, True)

    """
    # 1.4, visualize mock data
    """
    gbl_data = mock_data[('test', 'nan', str(measure_size), 'nan')].global_data
    if mpirank == 0:
        plt.plot(x, gbl_data[0])
        plt.savefig('testfield_mock_li.pdf')

    """
    # step 2, prepare pipeline and execute analysis
    """
    """
    # 2.1, ensemble likelihood
    """
    likelihood = EnsembleLikelihood(mock_data, mock_cov)  # initialize likelihood with measured info

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
    pipe = MultinestPipeline(simer, factory_list, likelihood, prior, simulation_size)
    pipe.random_type = 'free'
    pipe.sampling_controllers = {'n_iter_before_update': 1,
                                 'n_live_points': 400,
                                 'verbose': True,
                                 'resume': False}
    tmr = Timer()
    tmr.tick('test')
    results = pipe()  # run with pymultinest
    tmr.tock('test')
    if mpirank == 0:
        print('\n elapse time '+str(tmr.record['test'])+'\n')

    """
    # step 3, visualize (with corner package)
    """
    if mpirank == 0:
        samples = results['samples']
        for i in range(len(pipe.active_parameters)):  # convert variables into parameters
            low, high = pipe.active_ranges[pipe.active_parameters[i]]
            for j in range(samples.shape[0]):
                samples[j, i] = unity_mapper(samples[j, i], low, high)
        # corner plot
        corner.corner(samples[:, :len(pipe.active_parameters)],
                      range=[0.99]*len(pipe.active_parameters),
                      quantiles=[0.02, 0.5, 0.98],
                      labels=pipe.active_parameters,
                      show_titles=True,
                      title_kwargs={"fontsize": 20},
                      color='steelblue',
                      truths=truths,
                      truth_color='firebrick',
                      plot_contours=True,
                      hist_kwargs={'linewidth': 2},
                      label_kwargs={'fontsize': 20})
        plt.savefig('testfield_posterior_li_multinest.pdf')


if __name__ == '__main__':
    testfield(10, 100)
