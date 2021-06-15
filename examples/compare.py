#!/usr/bin/env python

# Built-in imports
import os, sys, logging
from mpi4py import MPI
# External packages
import numpy as np
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
# IMAGINE
import imagine as img
import imagine.observables as img_obs
## WMAP field factories
from imagine.fields.hamx import BregLSA, BregLSAFactory
from imagine.fields.hamx import TEregYMW16, TEregYMW16Factory
from imagine.fields.hamx import CREAna, CREAnaFactory
from imagine.fields.hamx import BrndES, BrndESFactory
from imagine.fields.hamx import BregCart, BregCartFactory
from imagine.fields.hamx import TEregUnif, TEregUnifFactory

from imagine.fields.basic_fields import ConstantThermalElectrons
from imagine.fields.basic_fields import ConstantMagneticField
from imagine.fields.basic_fields import RandomThermalElectrons

matplotlib.use('Agg')

# Sets up MPI variables
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

timer = img.tools.Timer()


def msg(txt, banner=True):
    if mpirank == 0:
        if banner:
            print('\n')
            print('-' * 60)
        print(txt, flush=True)
        if banner:
            print('-' * 60, flush=True)
    else:
        print('', flush=True, end='')  # Flushes STDOUT


def prepare_mock_obs_data(b0=3, psi0=27, rms=4, err=0.01, nside=2):
    """
    Prepares fake total intensity and Faraday depth data

    Parameters
    ----------
    b0, psi0 : float
        "True values" in the WMAP model

    Returns
    -------
    mock_data : imagine.observables.observables_dict.Measurements
        Mock Measurements
    mock_cov :  imagine.observables.observables_dict.Covariances
        Mock Covariances
    """
    ## Sets the resolution
    size = 12 * nside ** 2

    # Generates the fake datasets
    dm_dset = img_obs.DispersionMeasureHEALPixDataset(data=np.empty(size) * u.pc / u.cm ** 3)
    fd_dset = img_obs.FaradayDepthHEALPixDataset(data=np.empty(size) * u.rad / u.m ** 2)

    # Appends them to an Observables Dictionary
    trigger = img_obs.Measurements(dm_dset, fd_dset)

    # Prepares the Hammurabi simmulator for the mock generation

    grid = img.fields.UniformGrid(box=(100, 100, 60), resolution=(10, 10, 6))
    b_const = ConstantMagneticField(parameters={'Bx': 1e-6 * u.Gauss, 'By': 1e-6 * u.Gauss, 'Bz': 1e-6 * u.Gauss},
                                    grid=grid)
    ne_const = ConstantThermalElectrons(parameters={'ne': 1 * u.cm ** (-3)}, grid=grid)
    ne_rand = RandomThermalElectrons(parameters={'mean': 1 * u.cm ** (-3), 'std': 1 * u.cm ** (-3),
                                                 'min_ne': 0.01 * u.cm ** (-3)}, grid=grid)
    b_const_hamx = BregCart(parameters={'bx': 1e-6 * u.Gauss, 'by': 1e-6 * u.Gauss, 'bz': 1e-6 * u.Gauss})
    ne_const_hamx = TEregUnif(parameters={'n0': 1 * u.cm ** (-3), 'r0': 1 * u.pc})

    mock_generator = img.simulators.NiftyLOS(measurements=trigger, grid=grid, rtype='full_box', nside=nside)
    mock_generator_hamx = img.simulators.Hammurabi(measurements=trigger)
    ## Generate mock data (run hammurabi)
    import time

    s = time.time()
    outputs_hamx = mock_generator_hamx([b_const_hamx, ne_const_hamx])
    print(time.time() - s, 'hammurabi time', (time.time() - s) * 15, flush=True)
    s = time.time()
    outputs = mock_generator([b_const, ne_rand])
    print(time.time() - s, 'generator time', (time.time() - s) * 15, flush=True)

    ## Collect the outputs
    mockeddm = outputs[('dm', None, nside, None)].global_data[0]
    noisedm = err * mockeddm.mean()
    mockedRM = outputs[('fd', None, nside, None)].global_data[0]
    noiseRM = err * np.mean(np.abs(mockedRM))
    print(noiseRM, noisedm)

    datadm = mockeddm + np.random.normal(loc=0, scale=noisedm, size=size)
    dm_dset = img_obs.DispersionMeasureHEALPixDataset(data=datadm << u.pc / u.cm / u.cm / u.cm,
                                                error=noisedm << u.pc / u.cm / u.cm / u.cm,)

    dataRM = mockedRM + np.random.normal(loc=0, scale=noiseRM, size=size)
    fd_dset = img_obs.FaradayDepthHEALPixDataset(data=dataRM << u.rad / u.m / u.m,
                                                 error=noiseRM << u.rad / u.m / u.m)

    mock_data = img_obs.Measurements(dm_dset, fd_dset)

    return mock_data


def prepare_pipeline(pipeline_class=img.pipelines.MultinestPipeline,
                     sampling_controllers={}, ensemble_size=20,
                     run_directory='example_pipeline',
                     n_evals_report=50, nside=4,
                     true_pars={'bx': 1e-6, 'by': 1e-6, 'bz': 1e-6, 'ne': 1},
                     obs_err=0.01):
    # Creates run directory for storing the chains and log
    if mpirank == 0:
        os.makedirs(run_directory, exist_ok=True)
    comm.Barrier()

    # Creates the mock dataset based on "true" parameters provided
    msg('Generating mock data')
    mock_data = prepare_mock_obs_data(err=obs_err, nside=nside,
                                      **true_pars)
    msg('Preparing pipeline')

    # Setting up of the pipeline
    ## Use an ensemble to estimate the galactic variance
    likelihood = img.likelihoods.EnsembleLikelihoodDiagonal(mock_data)
    grid = img.fields.UniformGrid(box=(100, 100, 60), resolution=(10, 10, 6))
    b_const = ConstantMagneticField(parameters={'Bx': 1e-6 * u.Gauss, 'By': 1e-6 * u.Gauss, 'Bz': 1e-6 * u.Gauss},
                                    grid=grid)
    ne_const = ConstantThermalElectrons(parameters={'ne': 1 * u.cm ** (-3)}, grid=grid)
    ne_rand = RandomThermalElectrons(parameters={'mean': 1 * u.cm ** (-3), 'std': 1 * u.cm ** (-3),
                                                 'min_ne': 0.01 * u.cm ** (-3)}, grid=grid)

    b_const_factory = img.fields.FieldFactory(b_const)
    b_const_factory.active_parameters = ('Bx',)
    b_const_factory.priors = {'Bx': img.priors.GaussianPrior(0, 1e-6, unit= u.Gauss)}
    ne_const_factory = img.fields.FieldFactory(ne_const)
    ne_const_factory.priors = {'ne': img.priors.FlatPrior(0, 1.1, unit=u.cm ** (-3))}
    ne_const_factory.active_parameters = ('ne',)
    ne_rand_factory = img.fields.FieldFactory(ne_rand)
    ne_rand_factory.priors = {'mean': img.priors.FlatPrior(0, 1.1, unit=u.cm ** (-3))}
    ne_rand_factory.active_parameters = ('mean',)



    # Final Field factory list
    factory_list = [b_const_factory,
                    ne_rand_factory]

    # Prepares simulator

    # simulator = img.simulators.Hammurabi(measurements=mock_data)
    simulator = img.simulators.NiftyLOS(measurements=mock_data, grid=grid, rtype='full_box', nside=nside)
    # Prepares pipeline
    pipeline = pipeline_class(simulator=simulator,
                              show_progress_reports=True,
                              factory_list=factory_list,
                              n_evals_report=n_evals_report,
                              likelihood=likelihood,
                              ensemble_size=ensemble_size,
                              run_directory=run_directory)
    pipeline.sampling_controllers = sampling_controllers
    pipeline.save()

    return pipeline


def run_pipeline(pipeline, true_pars=None):
    # Runs!
    msg('Running the pipeline')
    timer.tick('pipeline')
    results = pipeline()
    total_time = timer.tock('pipeline')
    msg('\n\nFinished the run in {0:.2f}'.format(total_time), banner=False)
    comm.Barrier()

    if mpirank == 0:
        # Reports the evidence (to file)
        report_file = os.path.join(run_directory,
                                   'example_pipeline_results.txt')
        with open(report_file, 'w+') as f:
            f.write('log evidence: {}'.format(pipeline.log_evidence))
            f.write('log evidence error: {}'.format(pipeline.log_evidence_err))

        # Reports the posterior
        if true_pars is not None:
            f = pipeline.corner_plot(truths_dict=true_pars)

        f.savefig(os.path.join(run_directory, 'corner_plot_truth.pdf'))
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


def show_usage(cmd):
    print('IMAGINE example run\n')
    print('Usage: ')
    print('\t{} prepare\t   Prepares (or tests) an example Pipeline'.format(cmd))
    print('\t{} run\t   Runs an example Pipeline (preparing if necessary)'.format(cmd))
    exit()


if __name__ == '__main__':
    # Checks command line arguments
    cmd, args = sys.argv[0], sys.argv[1:]
    if len(args) == 0:
        show_usage(cmd)
    elif args[0] == 'prepare':
        prepare_only = True
    elif args[0] == 'run':
        prepare_only = False
    else:
        show_usage(cmd)

    msg('IMAGINE EXAMPLE RUN\n\n', banner=False)

    true_pars = {'b0': 6, 'psi0': 27, 'rms': 3}

    # Sets run directory name
    run_directory = os.path.join('runs', 'example_pipeline')

    if not os.path.isdir(run_directory) or not os.path.exists(run_directory + '/pipeline.pkl'):
        msg('Preparing Pipeline')
        pipeline = prepare_pipeline(
            ensemble_size=30,
            nside=8,
            sampling_controllers={'n_live_points': 50},
            run_directory=run_directory,
            true_pars=true_pars)
    else:
        msg('Loading Pipeline')
        pipeline = img.load_pipeline(run_directory)

    # Sets up logging
    logging.basicConfig(
        filename=os.path.join(run_directory, 'example_pipeline.log'),
        level=logging.INFO)

    # Tests and checks the runtime
    msg('Testing pipeline')
    if prepare_only:
        test_args = {'n_points': 3}
    else:
        test_args = {'n_points': 1, 'include_centre': False}

    pipeline.test(**test_args)

    # Runs the sampler
    if not prepare_only:
        run_pipeline(pipeline, true_pars=true_pars)
