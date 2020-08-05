"""
The basic elements of an IMAGINE pipeline

The present script exemplifies the usage of the basic features of IMAGINE and
the use of MPI parallelism.

This is the scripted version of the first IMAGINE tutorial. Before examining this
script, we strongly recommend reading the original tutorial either in the
documentation website or in the corresponding jupyter notebook in the tutorials
directory.
"""
import numpy as np
import astropy.units as u
import corner
import matplotlib
from scipy.stats import norm, truncnorm, truncexpon, pearsonr
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import imagine as img
import imagine.fields.test_field as testFields
from imagine.simulators.test_simulator import TestSimulator

from mpi4py import MPI


def prepare_prior_samples(n_samp, c, distr1, distr2):
    s1 = np.random.normal(0, 1, n_samp)
    s2 = np.random.normal(0, 1, n_samp)

    corr = np.eye(2)
    corr[0, 1] = corr[1, 0] = c

    A = np.linalg.cholesky(corr)

    s1, s2 = np.dot(A, np.asarray([s1, s2]))

    s1 = distr1.ppf(norm(0, 1).cdf(s1))
    s2 = distr2.ppf(norm(0, 1).cdf(s2))

    return s1, s2


def prepare_mock_dataset(a0, b0, psi0, size=10,
                         error=0.1, seed=233):
    r"""
    Prepares a mock dataset

    The model is characterised by:

    ..math::
        signal(x) = \left[1+\cos(psi*x)\right] \times \mathcal{G}(\mu=a_0,\sigma=b_0;seed=s)\,{\mu\rm G\,cm}^{-3} , \; x \in [0,2\pi]\,\rm kpc

    And the mock data is given by

    ..math::
        data(x) = signal(x) + noise(x)

    with

    ..math::
        noise(x) = \mathcal{G}(\mu=0,\sigma=e)

    where :math:`\mathcal{G}` is a Gaussian.

    Parameters
    ----------
    a0 : float
        Mean of the random magnetic field
    b0 : float
        Standard deviation of the random magnetic field
    size : int
        Number of datapoints
    error : float
        Error in the data
    seed : int
        Random seed that should be used

    Returns
    -------
    dataset : imagine.dataset
        An IMAGINE dataset object
    """
    # Sets the x coordinates of the observations
    x = np.linspace(0.01,2.*np.pi-0.01,size)
    # Sets the seed for signal field
    np.random.seed(seed)

    # Computes the signal
    signal = ((1+np.cos(psi0*x)) *
              np.random.normal(loc=a0, scale=b0, size=size))

    # Includes the error
    fd = signal + np.random.normal(loc=0.,scale=error,size=size)

    # Prepares a data dictionary
    data_dict = {'meas' : fd,
                 'err': np.ones_like(fd)*error,
                 'x': x,
                 'y': np.zeros_like(fd),
                 'z': np.zeros_like(fd)}

    fd_units = u.microgauss*u.cm**-3

    dataset = img.observables.TabularDataset(data_dict, name='test',
                                             data_column='meas',
                                             coordinates_type='cartesian',
                                             x_column='x', y_column='y', z_column='z',
                                             error_column='err', units=fd_units)
    return dataset


def plot_results(pipe, true_vals, output_file='test.pdf'):
    """
    Makes a cornerplot of the results and saves them to disk
    """
    samp = pipe.samples
    # Sets the levels to show 1, 2 and 3 sigma
    sigmas=np.array([1.,2.,3.])
    levels=1-np.exp(-0.5*sigmas*sigmas)

    # Visualize with a corner plot
    figure = corner.corner(np.vstack([samp.columns[0].value, samp.columns[1].value]).T,
                           range=[0.99]*len(pipe.active_parameters),
                           quantiles=[0.02, 0.5, 0.98],
                           labels=pipe.active_parameters,
                           show_titles=True,
                           title_kwargs={"fontsize": 12},
                           color='steelblue',
                           truths=true_vals,
                           truth_color='firebrick',
                           plot_contours=True,
                           hist_kwargs={'linewidth': 2},
                           label_kwargs={'fontsize': 10},
                           levels=levels)
    figure.savefig(output_file)


if __name__ == '__main__':


    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    mpisize = comm.Get_size()

    output_file_plot = 'correlation_pipeline_corner.pdf'
    output_text = 'correlation_pipeline_results.txt'

    # True values of the parameters
    a0 = 3.; b0 = 6; psi0 = 0.6
    # Generates the mock data
    mockDataset = prepare_mock_dataset(a0, b0, psi0)

    # Prepares Measurements and Covariances objects
    measurements = img.observables.Measurements()
    measurements.append(dataset=mockDataset)
    covariances = img.observables.Covariances()
    covariances.append(dataset=mockDataset)

    # Generates the grid
    one_d_grid = img.fields.UniformGrid(box=[[0, 2*np.pi]*u.kpc, [0, 0]*u.kpc, [0, 0]*u.kpc], resolution=[100, 1, 1])

    distr1 = truncnorm(loc=3, scale=2, a=- 3/2, b=3/2)
    distr2 = truncexpon(loc=0, scale=2, b=10)

    s1, s2 = prepare_prior_samples(1000, 0.8, distr1, distr2)

    print(s1.min(), s1.max())
    print(s2.min(), s2.max())
    print(s1)

    # Prepares the thermal electron field factory
    ne_factory = testFields.CosThermalElectronDensityFactory(grid=one_d_grid)

    ne_factory.default_parameters = {'beta': np.pi / 2 * u.rad,
                                     'gamma': np.pi / 2 * u.rad}

    ne_factory.active_parameters = ('a',)
    ne_factory.priors = {'a': img.priors.GeneralPrior(samples=s1*u.rad/u.kpc, interval=[0, 6]*u.rad/u.kpc), }

    # Prepares the random magnetic field factory
    B_factory = testFields.NaiveGaussianMagneticFieldFactory(grid=one_d_grid)
    B_factory.active_parameters = ('a0', 'b0')

    B_factory.priors = {'a0': img.priors.FlatPrior(interval=[-5, 5]*u.microgauss),
                        'b0': img.priors.GeneralPrior(samples=s2*u.microgauss, interval=[0, 14]*u.microgauss),
                        }

    corr_dict = {('a0', 'b0'): -0.3, ('b0', 'a'): True}

    # Sets the field factory list
    factory_list = [ne_factory, B_factory]

    # Initializes the simulator
    simer = TestSimulator(measurements)

    # Initializes the likelihood
    likelihood = img.likelihoods.EnsembleLikelihood(measurements, covariances)

    # Defines the pipeline using the UltraNest sampler, giving it the required elements
    pipeline = img.pipelines.UltranestPipeline(simulator=simer,
                                               factory_list=factory_list,
                                               likelihood=likelihood,
                                               ensemble_size=128,
                                               prior_correlations=corr_dict)
    pipeline.random_type = 'controllable'
    # Set some controller parameters that are specific to UltraNest.
    pipeline.sampling_controllers = {'max_ncalls': 1000,
                                     'Lepsilon': 0.1,
                                     'dlogz': 0.5,
                                     'min_num_live_points': 100}

    # RUNS THE PIPELINE
    results = pipeline()

    if mpirank == 0:
        # Reports the evidence (to file)
        with open(output_text,'w+') as f:
            f.write('log evidence: {}'.format( pipeline.log_evidence))
            f.write('log evidence error: {}'.format(pipeline.log_evidence_err))

        # Reports the posterior
        plot_results(pipeline, [a0,b0], output_file=output_file_plot)

        # Prints setup
        print('\nRC used:', img.rc)
        print('Seed used:', pipeline.master_seed)
        # Prints some results
        print('\nEvidence found:', pipeline.log_evidence, '±', pipeline.log_evidence_err)
        print('\nParameters summary:')
        for parameter in pipeline.active_parameters:
            print(parameter)
            constraints = pipeline.posterior_summary[parameter]
            for k in ['median','errup','errlo']:
                print('\t', k, constraints[k])
