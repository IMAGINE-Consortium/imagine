#%% Imports and settings

# Imagine
import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12Factory

# Utility
import os
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
import struct

# Directory paths
rundir    = 'runs/mockdata'
figpath   = 'figures/simulator_testing/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
Ndata = 100
observing_frequency = 90*MHz
dunit = u.K/u.kpc

#===========================================================================

def fill_imagine_dataset(data):
    fake_dset = img.observables.TabularDataset(data,
                                               name='average_los_brightness',
                                               frequency=observing_frequency,
                                               units=dunit,
                                               data_col='brightness',
                                               err_col='err',
                                               lat_col='lat',
                                               lon_col='lon')
    return img.observables.Measurements(fake_dset)

def produce_mock_data(field_list, mea, config, noise=0.01):
    """Runs the simulator once to produce a simulated dataset"""
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config)
    simulation = test_sim(field_list)
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    sim_brightness += np.random.normal(loc=0, scale=noise*sim_brightness, size=Ndata)*simulation[key].unit
    return sim_brightness

def randrange(minvalue,maxvalue,Nvalues):
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

def get_label_FB(Ndata):
    NF = int(0.2*Ndata) # 20% of all measurements are front measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB)
    return FB

#===========================================================================

# pipeline controller
def JF12pipeline():

    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 15*u.kpc
    ymax = 15*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata)
    y = randrange(-0.9*ymax,0.9*ymax,Ndata)
    z = randrange(-0.9*zmax,0.9*zmax,Ndata)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)

    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = [30,30,10]) # skipping x=y=0
    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                            parameters = {'scale_radius':10*u.kpc,
                                                         'scale_height':1*u.kpc,
                                                         'central_density':1e-5*u.cm**-3,
                                                         'spectral_index':-3})
    Bfield = WrappedJF12(grid=cartesian_grid)

    # Setup observing configuration
    observer = np.array([-8.5,0,0])*u.kpc
    dist_err = hIIdist/50000
    FB       = get_label_FB(Ndata)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}

    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config, noise=0.01)
    sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)

    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)

    # Setup field factories and their active parameters
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=cartesian_grid)
    B_factory   = WrappedJF12Factory(grid=cartesian_grid)
    """
    B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
    B_factory.active_parameters = ('b_arm_1','b_arm_2')
    B_factory.priors = {
    'b_arm_1':img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
    'b_arm_2':img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
    }
    """
    factory_list = [B_factory, CRE_factory]

    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                run_directory = rundir,
                                                factory_list  = factory_list,
                                                likelihood    = likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}

    # Run!
    results = pipeline()

    summary = pipeline.posterior_summary
    samples = pipeline.samples

    return samples, summary

# Clear previous pipeline
os.system("rm -r runs/mockdata/*")
JF12pipeline()
