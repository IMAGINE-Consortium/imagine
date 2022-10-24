#!/usr/bin/env python
"""
The basic elements of an IMAGINE pipeline

The present script exemplifies the usage of the basic features of IMAGINE and
the use of MPI parallelism.

This is the scripted version of the first IMAGINE tutorial. Before examining this
script, we strongly recommend reading the original tutorial either in the
documentation website or in the corresponding jupyter notebook in the tutorials
directory.

Run suggestion (from the IMAGINE path):
>>> mpirun -np 16 examples/basic_pipeline.py
"""

# Built-in imports
import os
import sys
import logging
from mpi4py import MPI
# External packages
import numpy as np
import astropy.units as u
import corner
import matplotlib
import matplotlib.pyplot as plt
# IMAGINE
import imagine as img
import imagine.observables as img_obs
import imagine.fields.test_field as testFields
from imagine.simulators.test_simulator import TestSimulator

matplotlib.use('Agg')
# Sets up MPI variables
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()
# Initializes timer object
timer = img.tools.Timer()


def prepare_mock_dataset(a0=3., b0=6., size=10,
                         error=0.1, seed=233):
    r"""
    Prepares a mock dataset

    The model is characterised by:

    ..math::
        signal(x) = \left[1+\cos(x)\right] \times \mathcal{G}(\mu=a_0,\sigma=b_0;seed=s)\,{\mu\rm G\,cm}^{-3} , \; x \in [0,2\pi]\,\rm kpc

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
    x = np.linspace(0.01, 2.*np.pi-0.01, size)
    # Sets the seed for signal field
    np.random.seed(seed)

    # Computes the signal
    signal = ((1+np.cos(x))
              * np.random.normal(loc=a0, scale=b0, size=size))

    # Includes the error
    fd = signal + np.random.normal(loc=0., scale=error, size=size)

    # Prepares a data dictionary
    data_dict = {'meas': u.Quantity(fd, u.microgauss*u.cm**-3),
                 'err': np.ones_like(fd)*error,
                 'x': x,
                 'y': np.zeros_like(fd),
                 'z': np.zeros_like(fd)}

    dataset = img.observables.TabularDataset(data_dict, name='test',
                                             data_col='meas',
                                             err_col='err')
    return dataset


def basic_pipeline_run(pipeline_class=img.pipelines.MultinestPipeline,
                       sampling_controllers={}, ensemble_size=48,
                       true_parameters={'a0': 3, 'b0': 6},
                       run_directory='basic_pipeline_run'):

    # Creates a directory for storing the chains and log
    chains_dir = os.path.join(run_directory, 'chains')
    if mpirank == 0:
        os.makedirs(chains_dir, exist_ok=True)
    comm.Barrier()

    # Sets up logging
    logging.basicConfig(
      filename=os.path.join(run_directory, 'basic_pipeline.log'),
      level=logging.INFO)

    output_file_plot = os.path.join(run_directory, 'corner_plot.pdf')
    output_text = os.path.join(run_directory, 'results.txt')

    # Generates the mock data
    mockDataset = prepare_mock_dataset(**true_parameters)

    # Prepares Measurements and Covariances objects
    measurements = img.observables.Measurements()
    measurements.append(dataset=mockDataset)
    covariances = img.observables.Covariances()
    covariances.append(dataset=mockDataset)

    # Generates the grid
    one_d_grid = img.fields.UniformGrid(box=[[0, 2*np.pi]*u.kpc,
                                             [0, 0]*u.kpc,
                                             [0, 0]*u.kpc],
                                        resolution=[100, 1, 1])

    # Prepares the thermal electron field factory
    ne_factory = testFields.CosThermalElectronDensityFactory(grid=one_d_grid)
    ne_factory.default_parameters = {'a': 1*u.rad/u.kpc,
                                     'beta':  np.pi/2*u.rad,
                                     'gamma': np.pi/2*u.rad}

    # Prepares the random magnetic field factory
    B_factory = testFields.NaiveGaussianMagneticFieldFactory(grid=one_d_grid)
    B_factory.active_parameters = ('a0', 'b0')
    B_factory.priors = {'a0': img.priors.FlatPrior(-5*u.microgauss, 5*u.microgauss),
                        'b0': img.priors.FlatPrior(2*u.microgauss, 10*u.microgauss)}

    # Sets the field factory list
    factory_list = [ne_factory, B_factory]

    # Initializes the simulator
    simer = TestSimulator(measurements)

    # Initializes the likelihood
    likelihood = img.likelihoods.EnsembleLikelihood(measurements, covariances)

    # Defines the pipeline using the UltraNest sampler, giving it the required elements
    pipeline = pipeline_class(simulator=simer,
                              factory_list=factory_list,
                              likelihood=likelihood,
                              ensemble_size=ensemble_size,
                              show_progress_reports=True,
                              chains_directory=chains_dir)
    pipeline.sampling_controllers = sampling_controllers

    # RUNS THE PIPELINE
    results = pipeline()

    if mpirank == 0:
        # Reports the evidence (to file)
        with open(output_text, 'w+') as f:
            f.write('log evidence: {}'.format(pipeline.log_evidence))
            f.write('log evidence error: {}'.format(pipeline.log_evidence_err))

        # Reports the posterior
        pipeline.corner_plot(truths_dict={
          'naive_gaussian_magnetic_field_a0': true_parameters['a0'],
          'naive_gaussian_magnetic_field_b0': true_parameters['b0']})
        plt.savefig(output_file_plot)

        # Prints setup
        print('\nRC used:', img.rc)
        print('Seed used:', pipeline.master_seed)
        # Prints some results
        print('\nEvidence found:', pipeline.log_evidence, '±', pipeline.log_evidence_err)
        print('\nParameters summary:')
        for parameter in pipeline.active_parameters:
            print(parameter)
            constraints = pipeline.posterior_summary[parameter]
            for k in ['median', 'errup', 'errlo']:
                print('\t', k, constraints[k])


if __name__ == '__main__':
    # Checks command line arguments
    cmd, args = sys.argv[0], sys.argv[1:]
    if len(args) == 0:
        choice = 'dynesty'
    elif args[0].lower() in ('multinest', 'ultranest', 'dynesty'):
        choice = args[0].lower()  # Supports any capitalization
    else:
        if mpirank == 0:
            print('\nIMAGINE basic pipeline example run\n')
            print('usage:  {} [SAMPLER]'.format(cmd))
            print('\noptional argument:')
            print('\tSAMPLER\t\tEither "MultiNest" (default), "UltraNest" or "Dynesty"\n')
        exit()

    if choice == 'multinest':
        #------ MultiNest -------
        # Sets run directory name
        run_directory = os.path.join('runs', 'basic_pipeline_run_multinest')
        # Sets up the sampler to be used
        pipeline_class = img.pipelines.MultinestPipeline
        # Set some controller parameters that are specific to MultiNest.
        sampling_controllers = {'n_live_points': 500, 'verbose': True}
        # Starts the run
        if mpirank == 0:
            print('\n\nRunning using MultiNest')
            timer.tick('MultiNest')

        basic_pipeline_run(run_directory=run_directory,
                           sampling_controllers=sampling_controllers,
                           pipeline_class=pipeline_class)
        if mpirank == 0:
            print('Finished. Ellapsed time:', timer.tock('MultiNest'))

    elif choice == 'ultranest':
        # ------ UltraNest -------
        # Sets run directory name
        run_directory = os.path.join('runs', 'basic_pipeline_run_ultranest')
        # Sets up the sampler to be used
        pipeline_class = img.pipelines.UltranestPipeline
        # Set some controller parameters that are specific to UltraNest.
        sampling_controllers = {'dlogz': 0.5,
                                'dKL': 0.1,
                                'min_num_live_points': 500,
                                'min_ess': 1000}
        # Starts the run
        if mpirank == 0:
            print('Running using UltraNest')
            timer.tick('UltraNest')

        basic_pipeline_run(run_directory=run_directory,
                           sampling_controllers=sampling_controllers,
                           pipeline_class=pipeline_class)
        if mpirank == 0:
            print('Finished. Ellapsed time:', timer.tock('UltraNest'))

    elif choice == 'dynesty':
        # ------ Dynesty -------
        img.rc['pipeline_distribute_ensemble'] = True

        # Sets run directory name
        run_directory = os.path.join('imagine_runs', 'basic_pipeline_run_dynesty')
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)
        # Sets up the sampler to be used
        pipeline_class = img.pipelines.DynestyPipeline
        # Set some controller parameters that are specific to Dynesty.
        sampling_controllers = {'nlive_init': 500,
                                'dlogz_init': 0.05,
                                'dynamic': True}
        # Starts the run
        if mpirank == 0:
            print('\n\nRunning using Dynesty')
            timer.tick('Dynesty')

        basic_pipeline_run(run_directory=run_directory,
                           sampling_controllers=sampling_controllers,
                           pipeline_class=pipeline_class)
        if mpirank == 0:
            print('Finished. Ellapsed time:', timer.tock('Dynesty'))
