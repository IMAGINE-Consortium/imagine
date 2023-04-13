import imagine as img
from imagine.simulators.rmlos import RMSimulator
from imagine.fields.library.jf12 import JF12Regular
from imagine.fields.library.ymw16 import YMW16

import os
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
import struct
import astropy.units as u

# Directory paths
rundir = 'runs/mockdata'
figpath = 'figures/simulator_testing/'
fieldpath = 'arrayfields/'
logdir = 'log/'

n_data = 100
dunit = u.rad/(u.m**2)

#===========================================================================


def fill_imagine_dataset(data):
    fake_dset = img.observables.TabularDataset(data,
                                               name='fd',
                                               units=dunit,
                                               data_col='fd',
                                               err_col='err',
                                               lat_col='lat',
                                               lon_col='lon')
    return img.observables.Measurements(fake_dset)


def produce_mock_data(field_list, mea, config, noise=0.01):
    """Runs the simulator once to produce a simulated dataset"""
    test_sim = RMSimulator(measurements=mea, sim_config=config)
    simulation = test_sim(field_list)
    key = ('fd', None, 'tab', None)
    sim_data = simulation[key].data[0] * simulation[key].unit
    sim_data += np.random.normal(loc=0, scale=noise*abs(sim_data), size=n_data)*simulation[key].unit
    return sim_data


def randrange(minvalue, maxvalue, Nvalues):
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

# ===========================================================================

# pipeline controller


def JF12pipeline():

    # Produce empty data format
    T = np.zeros(n_data)*dunit  # placeholder
    T_err = np.zeros(n_data)*dunit  # placeholder
    xmax = 15*u.kpc
    ymax = 15*u.kpc
    zmax = 2*u.kpc
    x = randrange(-0.9*xmax, 0.9*xmax, n_data)
    y = randrange(-0.9*ymax, 0.9*ymax, n_data)
    z = randrange(-0.9*zmax, 0.9*zmax, n_data)
    dist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc, y, z)
    fake_data = {'fd': T, 'err': T_err, 'lat': lat, 'lon': lon}
    mea = fill_imagine_dataset(data=fake_data)

    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                            resolution=[30, 30, 30])  # skipping x=y=0
    nth = img.fields.ConstantThermalElectrons(grid=cartesian_grid, parameters={'ne': 1*u.cm**(-3)})
    # nth = YMW16(grid=cartesian_grid, parameters={'ne': 1*u.cm**(-3)})
    bfield = JF12Regular(grid=cartesian_grid) # using default parameters

    # Setup observing configuration
    observer = np.array([-8.5, 0, 0])*u.kpc
    dist_err = dist/100000
    config = {'grid': cartesian_grid,
              'observer': observer,
              'dist': dist,
              'e_dist': dist_err,
              'lat': lat,
              'lon': lon,
              'FB': None}

    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[nth, bfield], mea=mea, config=config, noise=0.01)
    sim_data = {'fd': mock_data, 'err': mock_data/10, 'lat': config['lat'], 'lon': config['lon']}
    sim_mea = fill_imagine_dataset(sim_data)
    # Setup simulator
    los_simulator = RMSimulator(sim_mea, config)
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)

    # Setup field factories and their active parameters
    nth_factory = img.fields.FieldFactory(field_class=nth, grid=cartesian_grid)
    # B_factory = JF12Factory(grid=cartesian_grid)
    
    B_factory   = img.fields.FieldFactory(field_class = bfield, grid=config['grid'])
    B_factory.active_parameters = ('b_arm_1','b_arm_2')
    B_factory.priors = {
    'b_arm_1':img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
    'b_arm_2':img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
    }
    
    factory_list = [B_factory, nth_factory]
    # Setup final pipeline
    bpv = bfield.compute_field(12)
    nthv = nth.compute_field(12)  
    los_simulator.fields = {'thermal_electron_density': nthv, 'magnetic_field' : bpv}
    fake_dset = img.observables.TabularDataset({'fd': los_simulator.simulate(1, 1, 1, 1)},
                                               name='fd',
                                               units=dunit,
                                               data_col='fd',
                                               err_col=None)
    fake_dset = img.observables.Simulations(fake_dset)

    pipeline = img.pipelines.MultinestPipeline(simulator=los_simulator,
                                               run_directory=rundir,
                                               factory_list=factory_list,
                                               likelihood=likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 500}

    # Run!
    results = pipeline()

    summary = pipeline.posterior_summary
    samples = pipeline.samples

    return samples, summary


# Clear previous pipeline
os.system("rm -r runs/mockdata/*")
JF12pipeline()
