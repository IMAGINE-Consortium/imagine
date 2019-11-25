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
from mpi4py import MPI
from imagine import Simulations, Measurements, Covariances
from imagine import EnsembleLikelihood
from imagine import FlatPrior
from imagine import DynestyPipeline
from imagine import Hammurabi
from imagine import BregLSA
from imagine import BregLSAFactory
#from imagine import BrndES
#from imagine import BrndESFactory
from imagine import CREAna
from imagine import CREAnaFactory
from imagine import TEregYMW16
from imagine import TEregYMW16Factory
from imagine.tools.covariance_estimator import oas_mcov
from imagine.tools.mpi_helper import mpi_mean, mpi_eye

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

"""
# visualize posterior
import corner
import matplotlib
from imagine.tools.carrier_mapper import unity_mapper
matplotlib.use('Agg')
"""

def lsa_errprop():
    #log.basicConfig(filename='imagine.log', level=log.DEBUG)
    
    """
    only LSA regular magnetic field model in test, @ 23GHz
    Faraday rotation provided by YMW16 thermal electron model
    full LSA parameter set {b0, psi0, psi1, chi0}
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
        # BregLSA field
        paramlist = {'b0': b0_var[i], 'psi0': psi0_var[i], 'psi1': psi1_var[i], 'chi0': chi0_var[i]}  # inactive parameters at default
        breg_lsa = BregLSA(paramlist, 1)
        # CREAna field
        paramlist = {'alpha': alpha_var[i], 'beta': 0.0, 'theta': 0.0,
                     'r0': r0_var[i], 'z0': z0_var[i],
                     'E0': 20.6, 'j0': 0.0217}  # inactive parameters at default
        cre_ana = CREAna(paramlist, 1)
        # TEregYMW16 field
        tereg_ymw16 = TEregYMW16(dict(), 1)
        # collect mock data and covariance
        outputs = mocker([breg_lsa, cre_ana, tereg_ymw16])
        mock_ensemble.append(('sync', '23', str(mea_nside), 'I'), outputs[('sync', '23', str(mea_nside), 'I')])
    # collect mean and cov from simulated results
    mock_data = Measurements()
    mock_cov = Covariances()
    mean, cov = oas_mcov(mock_ensemble[('sync', '23', str(mea_nside), 'I')].data)
    mock_data.append(('sync', '23', str(mea_nside), 'I'), mean)
    mock_cov.append(('sync', '23', str(mea_nside), 'I'), cov)

    """
    # step 2, prepare pipeline and execute analysis
    """
    likelihood = EnsembleLikelihood(mock_data, mock_cov)

    breg_factory = BregLSAFactory(active_parameters=('b0', 'psi0', 'psi1', 'chi0'))
    breg_factory.parameter_ranges = {'b0': (0., 10.), 'psi0': (0., 50.), 'psi1': (0., 2.), 'chi0': (0., 50.)}
    cre_factory = CREAnaFactory(active_parameters=('alpha', 'r0', 'z0'))
    cre_factory.parameter_ranges = {'alpha': (1., 5.), 'r0': (1., 10.), 'z0': (0.1, 5.)}
    tereg_factory = TEregYMW16Factory()
    factory_list = [breg_factory, cre_factory, tereg_factory]

    prior = FlatPrior()

    simer = Hammurabi(measurements=mock_data, xml_path=xmlpath)

    ensemble_size = 1
    pipe = DynestyPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_type = 'free'
    pipe.sampling_controllers = {'nlive': 4000}
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
    """

def lsa_errfix():
    #log.basicConfig(filename='imagine.log', level=log.DEBUG)
    
    """
    only LSA regular magnetic field model in test, @ 23GHz
    Faraday rotation provided by YMW16 thermal electron model
    full LSA parameter set {b0, psi0, psi1, chi0}
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
    error = 0.1  # theoretical raltive uncertainty for each (active) parameter
    mocker = Hammurabi(measurements=trigger, xml_path=xmlpath)
    # start simulation
    # BregLSA field
    paramlist = {'b0': true_b0, 'psi0': true_psi0, 'psi1': true_psi1, 'chi0': true_chi0}  # inactive parameters at default
    breg_lsa = BregLSA(paramlist, 1)
    # CREAna field
    paramlist = {'alpha': true_alpha, 'beta': 0.0, 'theta': 0.0,
                 'r0': true_r0, 'z0': true_z0,
                 'E0': 20.6, 'j0': 0.0217}  # inactive parameters at default
    cre_ana = CREAna(paramlist, 1)
    # TEregYMW16 field
    tereg_ymw16 = TEregYMW16(dict(), 1)
    # collect mock data and covariance
    outputs = mocker([breg_lsa, cre_ana, tereg_ymw16])
    Imap = outputs[('sync', '23', str(mea_nside), 'I')].local_data
    # collect mean and cov from simulated results
    mock_data = Measurements()
    mock_cov = Covariances()
    mock_data.append(('sync', '23', str(mea_nside), 'I'), Imap)
    mock_cov.append(('sync', '23', str(mea_nside), 'I'), (error**2*(mpi_mean(Imap))**2)*mpi_eye(mea_pix))

    """
    # step 2, prepare pipeline and execute analysis
    """
    likelihood = EnsembleLikelihood(mock_data, mock_cov)

    breg_factory = BregLSAFactory(active_parameters=('b0', 'psi0', 'psi1', 'chi0'))
    breg_factory.parameter_ranges = {'b0': (0., 10.), 'psi0': (0., 50.), 'psi1': (0., 2.), 'chi0': (0., 50.)}
    cre_factory = CREAnaFactory(active_parameters=('alpha', 'r0', 'z0'))
    cre_factory.parameter_ranges = {'alpha': (1., 5.), 'r0': (1., 10.), 'z0': (0.1, 5.)}
    tereg_factory = TEregYMW16Factory()
    factory_list = [breg_factory, cre_factory, tereg_factory]

    prior = FlatPrior()

    simer = Hammurabi(measurements=mock_data, xml_path=xmlpath)

    ensemble_size = 1
    pipe = DynestyPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_type = 'free'
    pipe.sampling_controllers = {'nlive': 4000}
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
    """

if __name__ == '__main__':
    lsa_errprop()
    #lsa_errfix()
