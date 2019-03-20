"""
mock data generator
for WMAP + analytic CRE + YMW16
mask out l<60 + 4 loops

frequency 23 GHz
synchrotron Stockes Q, U
North pole
"""

import numpy as np
import healpy as hp
import logging as log

from imagine.observables.observable_dict import Masks
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

from imagine.tools.covariance_estimator import oas_cov


# mask loops and latitude
def mask_map(_nside, _freq):
    """
    return mask map dictionary for synchrotron Q, U
    at given nside and frequency
    """
    msk_map = np.zeros(hp.nside2npix(_nside))
    for _ipix in range(hp.nside2npix(_nside)):
        l,b = hp.pix2ang(_nside,_ipix,lonlat=True)
        R = np.pi/180.
        msk_map[_ipix] = 1
        L = [329,100,124,315]
        B = [17.5,-32.5,15.5,48.5]
        D = [116,91,65,39.5]
        #LOOP I
        if( np.arccos(np.sin(b*R)*np.sin(B[0]*R)+np.cos(b*R)*np.cos(B[0]*R)*np.cos(l*R-L[0]*R))<0.5*D[0]*R ):
            msk_map[_ipix] = 0
        #LOOP II
        elif( np.arccos(np.sin(b*R)*np.sin(B[1]*R)+np.cos(b*R)*np.cos(B[1]*R)*np.cos(l*R-L[1]*R))<0.5*D[1]*R ):
            msk_map[_ipix] = 0
        #LOOP III
        elif( np.arccos(np.sin(b*R)*np.sin(B[2]*R)+np.cos(b*R)*np.cos(B[2]*R)*np.cos(l*R-L[2]*R))<0.5*D[2]*R ):
            msk_map[_ipix] = 0
        #LOOP IV
        elif( np.arccos(np.sin(b*R)*np.sin(B[3]*R)+np.cos(b*R)*np.cos(B[3]*R)*np.cos(l*R-L[3]*R))<0.5*D[3]*R ):
            msk_map[_ipix] = 0
        #STRIPE
        elif(abs(b)<60.):
            msk_map[_ipix] = 0
    msk_dict = Masks()
    msk_dict.append(('sync', str(_freq), str(_nside), 'Q'), np.vstack([msk_map]))
    msk_dict.append(('sync', str(_freq), str(_nside), 'U'), np.vstack([msk_map]))
    return msk_dict


def mock_errprop(_nside, _freq):
    """
    return masked mock synchrotron Q, U
    error propagated from theoretical uncertainties
    """
    # hammurabi parameter base file
    xmlpath = './params_masked_regular.xml'
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
    mocksize = 100  # ensemble of mock data
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
    for i in range(mocksize):  # get one realization each time
        # BregWMAP field
        paramlist = {'b0': b0_var[i], 'psi0': psi0_var[i], 'psi1': psi1_var[i], 'chi0': chi0_var[i]}
        breg_wmap = BregWMAP(paramlist, 1)
        # CREAna field
        paramlist = {'alpha': alpha_var[i], 'beta': 0.0, 'theta': 0.0,
                     'r0': r0_var[i], 'z0': z0_var[i],
                     'E0': 20.6, 'j0': 0.0217}
        cre_ana = CREAna(paramlist, 1)
        # FEregYMW16 field
        paramlist = dict()
        fereg_ymw16 = FEregYMW16(paramlist, 1)
        # collect mock data and covariance
        outputs = mocker([breg_wmap, cre_ana, fereg_ymw16])
        mock_raw_q[i, :] = outputs[('sync', str(_freq), str(_nside), 'Q')].to_global_data()
        mock_raw_u[i, :] = outputs[('sync', str(_freq), str(_nside), 'U')].to_global_data()
    # collect mean and cov from simulated results
    sim_data = Simulations()
    mock_data = Measurements()
    mock_cov = Covariances()
    
    sim_data.append(('sync', str(_freq), str(_nside), 'Q'), mock_raw_q)
    sim_data.append(('sync', str(_freq), str(_nside), 'U'), mock_raw_u)
    mock_mask = mask_map(_nside, _freq)
    sim_data.apply_mask(mock_mask)
    for key in sim_data.keys():
        mock_data.append(key, np.vstack([(sim_data[key].to_global_data())[np.random.randint(0, mocksize)]]), True)
        mock_cov.append(key, oas_cov(sim_data[key].to_global_data()), True)
    return mock_data, mock_cov


def mock_errfix(_nside, _freq):
    """
    return masked mock synchrotron Q, U
    error fixed
    """
    # hammurabi parameter base file
    xmlpath = './params_masked_regular.xml'
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
    # BregWMAP field
    paramlist = {'b0': true_b0, 'psi0': true_psi0, 'psi1': true_psi1, 'chi0': true_chi0}
    breg_wmap = BregWMAP(paramlist, 1)
    # CREAna field
    paramlist = {'alpha': true_alpha, 'beta': 0.0, 'theta': 0.0,
                 'r0': true_r0, 'z0': true_z0,
                 'E0': 20.6, 'j0': 0.0217}
    cre_ana = CREAna(paramlist, 1)
    # FEregYMW16 field
    paramlist = dict()
    fereg_ymw16 = FEregYMW16(paramlist, 1)
    # collect mock data and covariance
    outputs = mocker([breg_wmap, cre_ana, fereg_ymw16])
    mock_raw_q = outputs[('sync', str(_freq), str(_nside), 'Q')]
    mock_raw_u = outputs[('sync', str(_freq), str(_nside), 'U')]
    # collect mean and cov from simulated results
    mock_data = Measurements()
    mock_cov = Covariances()
    
    mock_data.append(('sync', str(_freq), str(_nside), 'Q'), mock_raw_q)
    mock_data.append(('sync', str(_freq), str(_nside), 'U'), mock_raw_u)
    mock_mask = mask_map(_nside, _freq)
    mock_data.apply_mask(mock_mask)
    for key in mock_data.keys():
        mock_cov.append(key, (error**2*(np.std(mock_raw_q.to_global_data()))**2)*np.eye(int(key[2])), True)
    return mock_data, mock_cov


def main():
    #log.basicConfig(filename='imagine.log', level=log.DEBUG)
    
    nside = 4
    freq = 23
    
    mock_data, mock_cov = mock_errprop(nside, freq)
    mock_mask = mask_map(nside, freq)

    # using masked mock data/covariance
    # apply_mock will ignore masked input since mismatch in keys
    #likelihood = EnsembleLikelihood(mock_data, mock_cov, mock_mask)
    likelihood = SimpleLikelihood(mock_data, mock_cov, mock_mask)

    breg_factory = BregWMAPFactory(active_parameters=('b0', 'psi0', 'psi1', 'chi0'))
    breg_factory.parameter_ranges = {'b0': (0., 10.), 'psi0': (0., 50.), 'psi1': (0., 2.), 'chi0': (0., 50.)}
    cre_factory = CREAnaFactory(active_parameters=('alpha', 'r0', 'z0'))
    cre_factory.parameter_ranges = {'alpha': (1., 5.), 'r0': (1., 10.), 'z0': (0.1, 5.)}
    fereg_factory = FEregYMW16Factory()
    factory_list = [breg_factory, cre_factory, fereg_factory]

    prior = FlatPrior()

    xmlpath = './params_masked_regular.xml'
    # only for triggering simulator
    # since we use masked mock_data/covariance
    # if use masked input, outputs from simulator will not be masked due to mismatch in keys
    x = np.zeros((1, 12*nside**2))
    trigger = Measurements()
    trigger.append(('sync', str(freq), str(nside), 'Q'), x)
    trigger.append(('sync', str(freq), str(nside), 'U'), x)
    simer = Hammurabi(measurements=trigger, xml_path=xmlpath)

    ensemble_size = 1
    pipe = MultinestPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_type = 'free'
    pipe.sampling_controllers = {'resume': False, 'verbose': True, 'n_live_points': 4000}
    results = pipe()

    # saving results
    samples = results['samples']
    np.savetxt('posterior_masked_regular.txt', samples)

if __name__ == '__main__':
    main()
