#!/usr/bin/env python

# Built-in imports
import os
import sys
import logging
from mpi4py import MPI
# External packages
import numpy as np
import healpy as hp
import astropy.units as u
import corner
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

matplotlib.use('Agg')
# Sets up MPI variables
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

timer = img.tools.Timer()

def msg(txt, banner=True):
    if mpirank==0:
        if banner:
            print('\n')
            print('-'*60)
        print(txt, flush=True)
        if banner:
            print('-'*60, flush=True)
    else:
        print('', flush=True, end='') # Flushes STDOUT

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
    size = 12*nside**2

    # Generates the fake datasets
    sync_dset = img_obs.SynchrotronHEALPixDataset(data=np.empty(size)*u.K,
                                                  frequency=23*u.GHz, typ='I')
    fd_dset = img_obs.FaradayDepthHEALPixDataset(data=np.empty(size)*u.rad/u.m**2)

    # Appends them to an Observables Dictionary
    trigger = img_obs.Measurements()
    trigger.append(dataset=sync_dset)
    trigger.append(dataset=fd_dset)

    # Prepares the Hammurabi simmulator for the mock generation
    mock_generator = img.simulators.Hammurabi(measurements=trigger)

    # BregLSA field
    breg_lsa = BregLSA(parameters={'b0': b0, 'psi0': psi0, 'psi1': 0.9, 'chi0': 25.0})
    # CREAna field
    cre_ana = CREAna(parameters={'alpha': 3.0, 'beta': 0.0, 'theta': 0.0,
                                 'r0': 5.0, 'z0': 1.0,
                                 'E0': 20.6, 'j0': 0.0217})
    # TEregYMW16 field
    tereg_ymw16 = TEregYMW16(parameters={})
    ## Random field
    brnd_es = BrndES(parameters={'rms': rms, 'k0': 0.5, 'a0': 1.7,
                                 'k1': 0.5, 'a1': 0.0,
                                 'rho': 0.5, 'r0': 8., 'z0': 1.},
                     grid_nx=25, grid_ny=25, grid_nz=15)

    ## Generate mock data (run hammurabi)
    outputs = mock_generator([breg_lsa, brnd_es, cre_ana, tereg_ymw16])

    ## Collect the outputs
    mockedI = outputs[('sync', 23., nside, 'I')].global_data[0]
    mockedRM = outputs[('fd', None, nside, None)].global_data[0]
    dm=np.mean(mockedI)
    dv=np.std(mockedI)

    ## Add some noise that's just proportional to the average sync I by the factor err
    dataI = (mockedI + np.random.normal(loc=0, scale=err*dm, size=size)) << u.K
    errorI = ((err*dm)**2) << u.K
    sync_dset = img_obs.SynchrotronHEALPixDataset(data=dataI, error=errorI,
                                                  frequency=23*u.GHz, typ='I')
    ## Just 0.01*50 rad/m^2 of error for noise.
    dataRM = (mockedRM + np.random.normal(loc=0, scale=err*50,
                                          size=12*nside**2))*u.rad/u.m/u.m
    errorRM = ((err*50.)**2) << u.rad/u.m**2
    fd_dset = img_obs.FaradayDepthHEALPixDataset(data=dataRM, error=errorRM)

    mock_data = img_obs.Measurements()
    mock_data.append(dataset=sync_dset)
    mock_data.append(dataset=fd_dset)

    mock_cov = img_obs.Covariances()
    mock_cov.append(dataset=sync_dset)
    mock_cov.append(dataset=fd_dset)

    return mock_data, mock_cov

def prepare_pipeline(pipeline_class=img.pipelines.MultinestPipeline,
                     sampling_controllers={}, ensemble_size=10,
                     run_directory='example_pipeline',
                     n_evals_report=50, nside=4,
                     true_pars={'b0': 3, 'psi0': 27, 'rms': 4},
                     obs_err=0.01):

    # Creates run directory for storing the chains and log
    if mpirank==0:
        os.makedirs(run_directory, exist_ok=True)
    comm.Barrier()

    # Creates the mock dataset based on "true" parameters provided
    msg('Generating mock data')
    mock_data, mock_cov = prepare_mock_obs_data(err=obs_err, nside=nside,
                                                **true_pars)

    msg('Preparing pipeline')

    # Setting up of the pipeline
    ## Use an ensemble to estimate the galactic variance
    likelihood = img.likelihoods.EnsembleLikelihood(mock_data, mock_cov)

    ## WMAP B-field, vary only b0 and psi0
    breg_factory = BregLSAFactory()
    breg_factory.priors = {'b0':  img.priors.FlatPrior(xmin=2., xmax=8.),
                           'psi0': img.priors.FlatPrior(xmin=0., xmax=50.)}
    breg_factory.active_parameters = ('b0', 'psi0')
    ## Random B-field, vary only RMS amplitude
    brnd_factory = BrndESFactory(grid_nx=25, grid_ny=25, grid_nz=15)
    brnd_factory.active_parameters = ('rms',)
    brnd_factory.priors = {'rms': img.priors.FlatPrior(xmin=2., xmax=8.)}
    brnd_factory.default_parameters = {'k0': 0.5, 'a0': 1.7, 'k1': 0.5,
                                       'a1': 0.0, 'rho': 0.5, 'r0': 8., 'z0': 1.}
    ## Fixed CR model
    cre_factory = CREAnaFactory()
    ## Fixed FE model
    fereg_factory = TEregYMW16Factory()

    # Final Field factory list
    factory_list = [breg_factory, brnd_factory, cre_factory, fereg_factory]

    # Prepares simulator
    simulator = img.simulators.Hammurabi(measurements=mock_data)

    # Prepares pipeline
    pipeline = pipeline_class(simulator=simulator,
                              show_progress_reports=True,
                              factory_list=factory_list,
                              n_evals_report = n_evals_report,
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
    results=pipeline()
    total_time = timer.tock('pipeline')
    msg('\n\nFinished the run in {0:.2f}'.format(total_time), banner=False)
    comm.Barrier()

    if mpirank == 0:
        # Reports the evidence (to file)
        report_file=os.path.join(run_directory,
                            'example_pipeline_results.txt')
        with open(report_file, 'w+') as f:
            f.write('log evidence: {}'.format( pipeline.log_evidence))
            f.write('log evidence error: {}'.format(pipeline.log_evidence_err))

        # Reports the posterior
        if true_pars is not None:
            f = pipeline.corner_plot(truths_dict={'breg_lsa_b0': true_pars['b0'],
                                                  'breg_lsa_psi0': true_pars['psi0'],
                                                  'brnd_ES': true_pars['rms']})

        f.savefig(os.path.join(run_directory,'corner_plot_truth.pdf'))
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

def show_usage(cmd):
    print('IMAGINE example run\n')
    print('Usage: ')
    print('\t{} prepare\t   Prepares (or tests) an example Pipeline'.format(cmd))
    print('\t{} run\t   Runs an example Pipeline (preparing if necessary'.format(cmd))
    exit()

if __name__ == '__main__':
    # Checks command line arguments
    cmd, args = sys.argv[0], sys.argv[1:]
    if len(args)==0:
        show_usage(cmd)
    elif args[0]=='prepare':
        prepare_only = True
    elif args[0]=='run':
        prepare_only = False
    else:
        show_usage(cmd)

    msg('IMAGINE EXAMPLE RUN\n\n', banner=False)

    true_pars={'b0': 6, 'psi0': 27, 'rms': 3}

    # Sets run directory name
    run_directory=os.path.join('runs','example_pipeline')

    if not os.path.isdir(run_directory):
        msg('Preparing Pipeline')
        pipeline = prepare_pipeline(
          ensemble_size=10,
          nside=8,
          sampling_controllers={ 'n_live_points': 1000},
          run_directory=run_directory,
          true_pars=true_pars)
    else:
        msg('Loading Pipeline')
        pipeline = img.load_pipeline(run_directory)

    # Sets up logging
    logging.basicConfig(
      filename=os.path.join(run_directory, 'example_pipeline.log'),
      level=logging.INFO)

    # Checks the runtime
    msg('Testing pipeline')
    timer.tick('likelihood')
    pipeline._likelihood_function([3,3,3])
    test_time = timer.tock('likelihood')
    msg('Single likelihood evaluation: {0:.2f} s'.format(test_time), banner=False)

    if not prepare_only:
        run_pipeline(pipeline, true_pars=true_pars)
