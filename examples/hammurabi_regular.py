"""
hammurabi + regular field example
"""

import numpy as np
import logging as log

from imagine.observables.observable_dict import Simulations, Measurements, Covariances
from imagine.likelihoods.ensemble_likelihood import EnsembleLikelihood
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

# visualize posterior
import corner
import matplotlib
from imagine.tools.carrier_mapper import unity_mapper
matplotlib.use('Agg')


def wmap():
    log.basicConfig(filename='imagine.log', level=log.DEBUG)
    
    """
    only WMAP regular magnetic field model in test, @ 23GHz
    Faraday rotation provided by YMW16 free electron model
    full WMAP parameter set {b0, psi0, psi1, chi0}
    """
    
    true_b0 = 6.0
    true_psi0 = 27.0
    true_psi1 = 0.9
    true_chi0 = 25.0
    mea_std = 1.0e-4
    mea_nside = 2
    mea_pix = 12*mea_nside**2
    truths = [true_b0, true_psi0, true_psi1, true_chi0]

    """
    # step 1, prepare mock data
    """

    """
    # 1.1, generate measurements
    mea_field = signal_field + noise_field
    """
    x = np.zeros((1, mea_pix))
    measuredict = Measurements()
    measuredict.append(('sync', '23', str(mea_nside), 'I'), x)  # only I map
    xmlpath = './template.xml'
    mocker = Hammurabi(measurements=measuredict, xml_path=xmlpath)
    # BregWMAP field
    paramlist = {'b0': 6.0, 'psi0': 27.0, 'psi1': 0.9, 'chi0': 25.0}
    breg_wmap = BregWMAP(paramlist, 1)
    # CREAna field
    paramlist = {'alpha': 3.0, 'beta': 0.0, 'theta': 0.0,
                 'r0': 5.0, 'z0': 1.0,
                 'E0': 20.6, 'j0': 0.0217}
    cre_ana = CREAna(paramlist, 1)
    # FEregYMW16 field
    paramlist = dict()
    fereg_ymw16 = FEregYMW16(paramlist, 1)
    # collect mock data and covariance
    outputs = mocker([breg_wmap, cre_ana, fereg_ymw16])
    mock_data = Measurements()
    mock_cov = Covariances()
    mea_cov = (mea_std**2) * np.eye(mea_pix)
    for key in outputs.keys():
        mock_data.append(key, outputs[key])
        mock_cov.append(key, mea_cov)

    """
    # 1.2, visualize mock data
    """
    #import healpy as hp
    #sync_i_raw = outputs[('sync','23',str(mea_nside),'I')].to_global_data()
    #hp.write_map('mock.fits',sync_i_raw)

    """
    # step 2, prepare pipeline and execute analysis
    """
    likelihood = EnsembleLikelihood(mock_data, mock_cov)

    breg_factory = BregWMAPFactory(active_parameters=('b0', 'psi0'))
    breg_factory.parameter_ranges = {'b0': (0., 10.), 'psi0': (0., 50.)}
    cre_factory = CREAnaFactory(active_parameters=('alpha',))
    cre_factory.parameter_ragnes = {'alpha': (1., 5.)}
    fereg_factory = FEregYMW16Factory()
    factory_list = [breg_factory, cre_factory, fereg_factory]

    prior = FlatPrior()

    simer = Hammurabi(measurements=mock_data, xml_path=xmlpath)

    ensemble_size = 2
    pipe = MultinestPipeline(simer, factory_list, likelihood, prior, ensemble_size)
    pipe.random_seed = 0
    pipe.sampling_controllers = {'n_live_points': 1000, 'resume': False, 'verbose': True}
    results = pipe()

    """
    # step 3, visualize (with corner package)
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
                  truths=[true_b0],
                  truth_color='firebrick',
                  plot_contours=True,
                  hist_kwargs={'linewidth': 2},
                  label_kwargs={'fontsize': 20})
    matplotlib.pyplot.savefig('posterior.pdf')


if __name__ == '__main__':
	wmap()
