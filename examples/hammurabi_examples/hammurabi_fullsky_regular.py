"""
hammurabi + regular field example

without random fields, ensemble likelihood will act like simple likelihood
in order to check if the pipeline works as expected
we first put theoretical uncertainties in mock models and with error propagation
the mock data is produced with covariance with controlled source
a well-designed Bayesian analysis should be able to recover the pre-defined uncertainties

observables'/raidal integration resolutions are not very important in current example,
we set radial resolution as 0.1kpc, which brings, at maximum
2.5% relative integration error in simulated observables
"""

import numpy as np
import logging as log

import mpi4py

from nifty5 import Field, RGSpace
from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood
from imagine.likelihoods.simple_likelihood import SimpleLikelihood
from imagine.priors.flat_prior import FlatPrior
from imagine.pipelines.dynesty_pipeline import DynestyPipeline
from imagine.pipelines.multinest_pipeline import MultinestPipeline

from imagine.simulators.hammurabi.hammurabi import Hammurabi
from imagine.fields.breg_wmap.hamx_field import BregWMAP
from imagine.fields.breg_wmap.hamx_factory import BregWMAPFactory
from imagine.fields.brnd_es.hamx_field import BrndES
from imagine.fields.brnd_es.hamx_factory import BrndESFactory
from imagine.fields.cre_analytic.hamx_field import CREAna
from imagine.fields.cre_analytic.hamx_factory import CREAnaFactory
from imagine.fields.fereg_ymw16.hamx_field import FEregYMW16
from imagine.fields.fereg_ymw16.hamx_factory import FEregYMW16Factory

from imagine.tools.covariance_estimator import oas_mcov

comm = mpi4py.MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

"""
# visualize posterior
import corner
import matplotlib
from imagine.tools.carrier_mapper import unity_mapper
matplotlib.use('Agg')
"""


def wmap_errprop():
    #log.basicConfig(filename='imagine.log', level=log.DEBUG)
    
    """
    only WMAP regular magnetic field model in test, @ 23GHz
    Faraday rotation provided by YMW16 free electron model
    full WMAP parameter set {b0, psi0, psi1, chi0}
    """
    # hammurabi parameter base file
    xmlpath = './params_fullsky_regular.xml'
    
    # we take three active parameters
    true_b0 = 6.0
    true_psi0 = 27.0
    true_psi1 = 0.9
    true_chi0 = 25.
    true_alpha = 3.0
    true_r0 = 5.0
    true_z0 = 1.0
    
    mea_nside = 2  # observable Nside
    mea_pix = 12*mea_nside**2  # observable pixel number

    """
    # step 1, prepare mock data
    """
    x = np.zeros((1, mea_pix))  # only for triggering simulator
    trigger = Measurements()
    trigger.append(('sync', '23', str(mea_nside), 'I'), x)  # only I map
    # initialize simulator
    mocksize = 10  # ensemble of mock data (per node)
    error = 0.1  # theoretical raltive uncertainty for each (active) parameter
    mocker = Hammurabi(measurements=trigger, xml_path=xmlpath)
    # prepare theoretical uncertainty
    b0_var = np.random.normal(true_b0, error*true_b0, mocksize)
    psi0_var = np.random.normal(true_psi0, error*true_psi0, mocksize)
    psi1_var = np.random.normal(true_psi1, error*true_psi1, mocksize)
    chi0_var = np.random.normal(true_chi0, error*true_chi0, mocksize)
    alpha_var = np.random.normal(true_alpha, error*true_alpha, mocksize)
    r0_var = np.random.normal(true_r0, error*true_r0, mocksize)
    z0_var = np.random.normal(true_z0, error*true_z0, mocksize)
    mock_ensemble = Simulations()
    # start simulation
    for i in range(mocksize):  # get one realization each time
        # BregWMAP field
        paramlist = {'b0': b0_var[i], 'psi0': psi0_var[i], 'psi1': psi1_var[i], 'chi0': chi0_var[i]}  # inactive parameters at default
        breg_wmap = BregWMAP(paramlist, 1)
        # CREAna field
        paramlist = {'alpha': alpha_var[i], 'beta': 0.0, 'theta': 0.0,
                     'r0': r0_var[i], 'z0': z0_var[i],
                     'E0': 20.6, 'j0': 0.0217}  # inactive parameters at default
        cre_ana = CREAna(paramlist, 1)
        # FEregYMW16 field
        fereg_ymw16 = FEregYMW16(dict(), 1)
        # collect mock data and covariance
        outputs = mocker([breg_wmap, cre_ana, fereg_ymw16])
        mock_ensemble.append(('sync', '23', str(mea_nside), 'I'), outputs[('sync', '23', str(mea_nside), 'I')])
    # collect mean and cov from simulated results
    mock_data = Measurements()
    mock_cov = Covariances()
    mean, cov = oas_mcov(mock_ensemble[('sync', '23', str(mea_nside), 'I')])
    mock_data.append(('sync', '23', str(mea_nside), 'I'), mean)
    mock_cov.append(('sync', '23', str(mea_nside), 'I'), cov)

    """
    # step 2, prepare pipeline and execute analysis
    """
    #likelihood = EnsembleLikelihood(mock_data, mock_cov)
    likelihood = SimpleLikelihood(mock_data, mock_cov)

    breg_factory = BregWMAPFactory(active_parameters=('b0', 'psi0', 'psi1', 'chi0'))
    breg_factory.parameter_ranges = {'b0': (0., 10.), 'psi0': (0., 50.), 'psi1': (0., 2.), 'chi0': (0., 50.)}
    cre_factory = CREAnaFactory(active_parameters=('alpha', 'r0', 'z0'))
    cre_factory.parameter_ranges = {'alpha': (1., 5.), 'r0': (1., 10.), 'z0': (0.1, 5.)}
    fereg_factory = FEregYMW16Factory()
    factory_list = [breg_factory, cre_factory, fereg_factory]

    prior = FlatPrior()

    simer = Hammurabi(measurements=mock_data, xml_path=xmlpath)

    ensemble_size = 1
    pipe = MultinestPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_type = 'free'
    pipe.sampling_controllers = {'n_live_points': 4000, 'resume': False, 'verbose': True}
    results = pipe()

    """
    # step 3, visualize (with corner package)
    """
    if mpirank == 0:
        samples = results['samples']
        np.savetxt('posterior_fullsky_regular_errprop.txt', samples)
    """
    # screen printing
    print('\n evidence: %(logZ).1f +- %(logZerr).1f \n' % results)
    print('parameter values: \n')
    for name, col in zip(pipe.active_parameters, results['samples'].transpose()):
        print('%15s : %.3f +- %.3f \n' % (name, col.mean(), col.std()))

    # posterior plotting
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
                  truth_color='firebrick',
                  plot_contours=True,
                  hist_kwargs={'linewidth': 2},
                  label_kwargs={'fontsize': 20})
    matplotlib.pyplot.savefig('posterior_fullsky_regular_errprop.pdf')
    """

def wmap_errfix():
    #log.basicConfig(filename='imagine.log', level=log.DEBUG)
    
    """
    only WMAP regular magnetic field model in test, @ 23GHz
    Faraday rotation provided by YMW16 free electron model
    full WMAP parameter set {b0, psi0, psi1, chi0}
    """
    # hammurabi parameter base file
    xmlpath = './params_fullsky_regular.xml'
    
    # we take three active parameters
    true_b0 = 6.0
    true_psi0 = 27.0
    true_psi1 = 0.9
    true_chi0 = 25.
    true_alpha = 3.0
    true_r0 = 5.0
    true_z0 = 1.0
    truths = [true_b0, true_psi0, true_psi1, true_chi0, true_alpha, true_r0, true_z0]
    
    mea_nside = 2  # observable Nside
    mea_pix = 12*mea_nside**2  # observable pixel number

    """
    # step 1, prepare mock data
    """
    x = np.zeros((1, mea_pix))  # only for triggering simulator
    trigger = Measurements()
    trigger.append(('sync', '23', str(mea_nside), 'I'), x)  # only I map
    # initialize simulator
    error = 0.1  # theoretical raltive uncertainty for each (active) parameter
    mocker = Hammurabi(measurements=trigger, xml_path=xmlpath)
    # start simulation
    # BregWMAP field
    paramlist = {'b0': true_b0, 'psi0': true_psi0, 'psi1': true_psi1, 'chi0': true_chi0}  # inactive parameters at default
    breg_wmap = BregWMAP(paramlist, 1)
    # CREAna field
    paramlist = {'alpha': true_alpha, 'beta': 0.0, 'theta': 0.0,
                 'r0': true_r0, 'z0': true_z0,
                 'E0': 20.6, 'j0': 0.0217}  # inactive parameters at default
    cre_ana = CREAna(paramlist, 1)
    # FEregYMW16 field
    fereg_ymw16 = FEregYMW16(dict(), 1)
    # collect mock data and covariance
    outputs = mocker([breg_wmap, cre_ana, fereg_ymw16])
    Imap = outputs[('sync', '23', str(mea_nside), 'I')].local_data
    # collect mean and cov from simulated results
    mock_data = Measurements()
    mock_cov = Covariances()
    mock_data.append(('sync', '23', str(mea_nside), 'I'), Imap)
    mock_cov.append(('sync', '23', str(mea_nside), 'I'),
                    Field.from_global_data(RGSpace(shape=(mea_pix, mea_pix)), (error**2*(np.mean(Imap))**2)*np.eye(mea_pix)))

    """
    # step 2, prepare pipeline and execute analysis
    """
    #likelihood = EnsembleLikelihood(mock_data, mock_cov)
    likelihood = SimpleLikelihood(mock_data, mock_cov)

    breg_factory = BregWMAPFactory(active_parameters=('b0', 'psi0', 'psi1', 'chi0'))
    breg_factory.parameter_ranges = {'b0': (0., 10.), 'psi0': (0., 50.), 'psi1': (0., 2.), 'chi0': (0., 50.)}
    cre_factory = CREAnaFactory(active_parameters=('alpha', 'r0', 'z0'))
    cre_factory.parameter_ranges = {'alpha': (1., 5.), 'r0': (1., 10.), 'z0': (0.1, 5.)}
    fereg_factory = FEregYMW16Factory()
    factory_list = [breg_factory, cre_factory, fereg_factory]

    prior = FlatPrior()

    simer = Hammurabi(measurements=mock_data, xml_path=xmlpath)

    ensemble_size = 1
    pipe = MultinestPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_type = 'free'
    pipe.sampling_controllers = {'n_live_points': 4000, 'resume': False, 'verbose': True}
    results = pipe()

    """
    # step 3, visualize (with corner package)
    """
    if mpirank == 0:
        samples = results['samples']
        np.savetxt('posterior_fullsky_regular_errfix.txt', samples)
    """
    # screen printing
    print('\n evidence: %(logZ).1f +- %(logZerr).1f \n' % results)
    print('parameter values: \n')
    for name, col in zip(pipe.active_parameters, results['samples'].transpose()):
        print('%15s : %.3f +- %.3f \n' % (name, col.mean(), col.std()))

    # posterior plotting
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
                  truth_color='firebrick',
                  plot_contours=True,
                  hist_kwargs={'linewidth': 2},
                  label_kwargs={'fontsize': 20})
    matplotlib.pyplot.savefig('posterior_fullsky_regular_errfix.pdf')
    """

if __name__ == '__main__':
    #wmap_errprop()
    wmap_errfix()
