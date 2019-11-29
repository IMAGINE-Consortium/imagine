"""
mock data generator
with LSA + analytic CRE + YMW16
wiht iso-angular-sep mask

frequency 23 GHz
synchrotron Stockes Q, U
"""

import os
import numpy as np
import healpy as hp
import logging as log
from mpi4py import MPI
from imagine import Masks
from imagine import Simulations, Measurements, Covariances
from imagine import EnsembleLikelihood
from imagine import FlatPrior
from imagine import DynestyPipeline
from imagine import Hammurabi
from imagine import BregLSA
from imagine import BregLSAFactory
from imagine import CREAna
from imagine import CREAnaFactory
from imagine import TEregYMW16
from imagine import TEregYMW16Factory
from imagine.tools.covariance_estimator import oas_cov
from imagine.tools.mpi_helper import mpi_eye
from imagine.tools.timer import Timer


comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()


def mask_map_prod(_nside,_clon,_clat,_sep):
    """
    return a mask map array, in healpy RING ordering
    with iso-angular-separation-cut with respect to given cnetral galactic
    longitude and latitude
    
    Parameters
    ----------
    
    nside
        healpix Nside
        
    _clon
        central longitude, in degree
        
    _clat
        central latitude, in degree
        
    _sep
        angular separation, in degree
        
    Returns
    -------
    numpy.ndarray with bool data type (copied to all nodes)
    """
    _c = np.pi/180
    def hav(_theta):
        return 0.5-0.5*np.cos(_theta)
    tmp = np.ones(hp.nside2npix(_nside),dtype=bool)
    for _ipix in range(len(tmp)):
        lon,lat = hp.pix2ang(_nside,_ipix,lonlat=True)
        # iso-angle separation
        if((hav(np.fabs(_clat-lat)*_c)+np.cos(_clat*_c)*np.cos(lat*_c)*hav(np.fabs(_clon-lon)*_c))>hav(_sep*_c)):
            tmp[_ipix] = False
    if not mpirank:
        if os.path.isfile('hammurabi_mask.fits'):
            os.remove('hammurabi_mask.fits')
        hp.write_map('hammurabi_mask.fits',tmp)
    return tmp


def mock_errprop(_nside, _freq):
    """
    return masked mock synchrotron Q, U
    error propagated from theoretical uncertainties
    """
    # hammurabi parameter base file
    xmlpath = './params.xml'
    # active parameters
    true_b0 = 6.0
    true_psi0 = 27.0
    true_psi1 = 0.9
    true_chi0 = 25.
    true_alpha = 3.0
    true_r0 = 5.0
    true_z0 = 1.0
    #
    _npix = 12*_nside**2
    #
    x = np.zeros((1, _npix))  # only for triggering simulator
    trigger = Measurements()
    trigger.append(('sync', str(_freq), str(_nside), 'Q'), x)  # Q map
    trigger.append(('sync', str(_freq), str(_nside), 'U'), x)  # U map
    # initialize simulator
    mocksize = 20  # ensemble of mock data (per node)
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
    mock_raw_q = np.zeros((mocksize, _npix))
    mock_raw_u = np.zeros((mocksize, _npix))
    # start simulation
    np.random.seed(mpirank*10)
    for i in range(mocksize):  # get one realization each time
        # BregLSA field
        paramlist = {'b0': b0_var[i], 'psi0': psi0_var[i], 'psi1': psi1_var[i], 'chi0': chi0_var[i]}
        breg_lsa = BregLSA(paramlist, 1)
        # CREAna field
        paramlist = {'alpha': alpha_var[i], 'beta': 0.0, 'theta': 0.0,
                     'r0': r0_var[i], 'z0': z0_var[i],
                     'E0': 20.6, 'j0': 0.0217}
        cre_ana = CREAna(paramlist, 1)
        # TEregYMW16 field
        paramlist = dict()
        fereg_ymw16 = TEregYMW16(paramlist, 1)
        # collect mock data and covariance
        outputs = mocker([breg_lsa, cre_ana, fereg_ymw16])
        mock_raw_q[i, :] = outputs[('sync', str(_freq), str(_nside), 'Q')].data
        mock_raw_u[i, :] = outputs[('sync', str(_freq), str(_nside), 'U')].data
    # collect mean and cov from simulated results
    sim_data = Simulations()
    mock_data = Measurements()
    mock_cov = Covariances()
    mock_mask = Masks()
    
    sim_data.append(('sync', str(_freq), str(_nside), 'Q'), mock_raw_q)
    sim_data.append(('sync', str(_freq), str(_nside), 'U'), mock_raw_u)
    
    mask_map = mask_map_prod(_nside, 0, 90, 50)  # not parameterizing this
    mock_mask.append(('sync', str(_freq), str(_nside), 'Q'), np.vstack([mask_map]))
    mock_mask.append(('sync', str(_freq), str(_nside), 'U'), np.vstack([mask_map]))
    sim_data.apply_mask(mock_mask)
    for key in sim_data.keys():
        global_mock = np.vstack([(sim_data[key].data)[0]])
        comm.Bcast(global_mock, root=0)
        mock_data.append(key, global_mock, True)
        mock_cov.append(key, oas_cov(sim_data[key].data), True)
    return mock_data, mock_cov


def mock_errfix(_nside, _freq):
    """
    return masked mock synchrotron Q, U
    error fixed
    """
    # hammurabi parameter base file
    xmlpath = './params.xml'
    # active parameters
    true_b0 = 6.0
    true_psi0 = 27.0
    true_psi1 = 0.9
    true_chi0 = 25.
    true_alpha = 3.0
    true_r0 = 5.0
    true_z0 = 1.0
    #
    _npix = 12*_nside**2
    #
    x = np.zeros((1, _npix))  # only for triggering simulator
    trigger = Measurements()
    trigger.append(('sync', str(_freq), str(_nside), 'Q'), x)  # Q map
    trigger.append(('sync', str(_freq), str(_nside), 'U'), x)  # U map
    # initialize simulator
    error = 0.1
    mocker = Hammurabi(measurements=trigger, xml_path=xmlpath)
    # start simulation
    # BregLSA field
    paramlist = {'b0': true_b0, 'psi0': true_psi0, 'psi1': true_psi1, 'chi0': true_chi0}
    breg_lsa = BregLSA(paramlist, 1)
    # CREAna field
    paramlist = {'alpha': true_alpha, 'beta': 0.0, 'theta': 0.0,
                 'r0': true_r0, 'z0': true_z0,
                 'E0': 20.6, 'j0': 0.0217}
    cre_ana = CREAna(paramlist, 1)
    # TEregYMW16 field
    paramlist = dict()
    fereg_ymw16 = TEregYMW16(paramlist, 1)
    # collect mock data and covariance
    outputs = mocker([breg_lsa, cre_ana, fereg_ymw16])
    mock_raw_q = outputs[('sync', str(_freq), str(_nside), 'Q')].data
    mock_raw_u = outputs[('sync', str(_freq), str(_nside), 'U')].data
    # collect mean and cov from simulated results
    mock_data = Measurements()
    mock_cov = Covariances()
    mock_mask = Masks()
    
    mock_data.append(('sync', str(_freq), str(_nside), 'Q'), mock_raw_q)
    mock_data.append(('sync', str(_freq), str(_nside), 'U'), mock_raw_u)
    
    mask_map = mask_map_prod(_nside, 0, 90, 50)  # not parameterizing this
    mock_mask.append(('sync', str(_freq), str(_nside), 'Q'), np.vstack([mask_map]))
    mock_mask.append(('sync', str(_freq), str(_nside), 'U'), np.vstack([mask_map]))
    mock_data.apply_mask(mock_mask)
    for key in mock_data.keys():
        mock_cov.append(key, (error**2*(np.std(mock_raw_q))**2)*mpi_eye(int(key[2])), True)
    return mock_data, mock_cov


def main():
    #log.basicConfig(filename='imagine.log', level=log.DEBUG)
    
    nside = 2
    freq = 23
    
    mock_data, mock_cov = mock_errfix(nside, freq)
    mask_map = mask_map_prod(nside, 0, 90, 50)  # not parameterizing this
    mock_mask = Masks()
    mock_mask.append(('sync', str(freq), str(nside), 'Q'), np.vstack([mask_map]))
    mock_mask.append(('sync', str(freq), str(nside), 'U'), np.vstack([mask_map]))
    
    # using masked mock data/covariance
    # apply_mock will ignore masked input since mismatch in keys
    likelihood = EnsembleLikelihood(mock_data, mock_cov, mock_mask)

    breg_factory = BregLSAFactory(active_parameters=('b0',))  # set active parameters
    breg_factory.parameter_ranges = {'b0': (0., 10.)}
    
    cre_factory = CREAnaFactory(active_parameters=('alpha',))  # set active parameters
    cre_factory.parameter_ranges = {'alpha': (1., 5.)}
    
    fereg_factory = TEregYMW16Factory()
    factory_list = [breg_factory, cre_factory, fereg_factory]

    prior = FlatPrior()

    xmlpath = './params.xml'
    # only for triggering simulator
    # since we use masked mock_data/covariance
    # if use masked input, outputs from simulator will not be masked due to mismatch in keys
    x = np.zeros((1, 12*nside**2))
    trigger = Measurements()
    trigger.append(('sync', str(freq), str(nside), 'Q'), x)
    trigger.append(('sync', str(freq), str(nside), 'U'), x)
    simer = Hammurabi(measurements=trigger, xml_path=xmlpath)

    ensemble_size = 5
    pipe = DynestyPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_type = 'free'
    pipe.sampling_controllers = {'nlive': 400}
    
    tmr = Timer()
    tmr.tick('test')
    results = pipe()
    tmr.tock('test')
    if not mpirank:
        print('\n elapse time '+str(tmr.record['test'])+'\n')

    # saving results
    if mpirank == 0:
        samples = results['samples']
        np.savetxt('posterior_masked_regular.txt', samples)


if __name__ == '__main__':
    main()
