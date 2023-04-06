"""
This script is used to do performance testing of the galactic los synchrotron simulator
and the retrieval of model parameters from simulated datasets.



"""

#%% Imports and settings
#===================================================================================
# Imagine
import pandas as pd
import seaborn as sns
import astropy.units as u
import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12Factory
from imagine.fields.field_utility import FieldAdder
from imagine.fields.field_utility import ArrayMagneticField

# Utility
import os
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
import struct

# Directory paths
rundir = 'runs/mockdata'
figpath = 'figures/simulator_testing/'
fieldpath = 'arrayfields/'
logdir = 'log/'

# Gobal testing constants
MHz = 1e6 / u.s
Ndata = 100
observing_frequency = 90*MHz
dunit = u.K/u.kpc


#%% Code reduction functions
#===================================================================================

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
    test_sim = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config)
    simulation = test_sim(field_list)
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    sim_brightness += np.random.normal(loc=0, scale=noise*sim_brightness, size=Ndata)*simulation[key].unit
    return sim_brightness


def randrange(minvalue, maxvalue, Nvalues):
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue


def get_label_FB(Ndata):
    NF = int(0.2*Ndata)  # 20% of all measurements are front measurements
    F = np.full(shape=(NF), fill_value='F', dtype=str)
    B = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F, B))
    np.random.shuffle(FB)
    return FB


def load_JF12rnd(shape=(30, 30, 30, 3)):
    with open(fieldpath+"brnd.bin", "rb") as f:
        arr = f.read()
        arr = struct.unpack("d"*(len(arr)//8), arr[:])
        arr = np.asarray(arr).reshape(shape)
    return arr


#%% Pipeline controllers
#===================================================================================

# simulator testing
def test_simulator(Ndata, resolution=[30, 30, 30], keepinit=False):
    """Do simple speedtesting of the los-simulator"""

    # Produce empty data format
    T = np.zeros(Ndata)*dunit
    T_err = np.zeros(Ndata)*dunit
    lat = 90*np.linspace(-1, 1, Ndata)*u.deg
    lon = 360*np.linspace(0, 1, Ndata)*u.deg*300
    fake_data = {'brightness': T, 'err': T_err, 'lat': lat, 'lon': lon}
    mea = fill_imagine_dataset(data=fake_data)

    # Setup the Galactic field models
    box_size = 15*u.kpc
    cartesian_grid = img.fields.UniformGrid(box=[[-box_size, box_size],
                                                 [-box_size, box_size],
                                                 [-box_size, box_size]],
                                            resolution=resolution)
    cre = img.fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                                parameters={'density': 1.0e-7*u.cm**-3, 'spectral_index': -3.0})
    Bfield = img.fields.ConstantMagneticField(
        grid=cartesian_grid,
        parameters={'Bx': 6*u.microgauss,
                    'By': 0*u.microgauss,
                    'Bz': 0*u.microgauss})

    # Setup observing configuration
    observer = np.array([0, 0, 0])*u.kpc
    hIIdist = (box_size-2*u.kpc)*np.random.rand(Ndata) + 1*u.kpc  # uniform between [1, max-1] kpc
    dist_err = hIIdist/100000
    FB = get_label_FB(Ndata)
    config = {'grid': cartesian_grid,
              'observer': observer,
              'dist': hIIdist,
              'e_dist': dist_err,
              'lat': lat,
              'lon': lon,
              'FB': FB}

    # Do simulation
    start = perf_counter()
    test_sim = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config)
    inittime = perf_counter() - start
    simulation = test_sim(field_list=[cre, Bfield])
    stop = perf_counter()
    if keepinit:
        return stop - start, inittime
    else:
        return stop - start

# pipeline controller


def simple_pipeline(noise=0.1, fakemodel=False):
    """
    Test retrieval of correct cre spectral index for CONSTANT GMF and CRE density
    """

    # Produce empty data format
    T = np.zeros(Ndata)*dunit
    T_err = np.zeros(Ndata)*dunit
    lat = 90*np.linspace(-1, 1, Ndata)*u.deg
    lon = 360*np.linspace(0, 1, Ndata)*u.deg*300
    fake_data = {'brightness': T, 'err': T_err, 'lat': lat, 'lon': lon}
    mea = fill_imagine_dataset(data=fake_data)

    # Setup the Galactic field models
    box_size = 15*u.kpc
    cartesian_grid = img.fields.UniformGrid(box=[[-box_size, box_size],
                                                 [-box_size, box_size],
                                                 [-box_size, box_size]],
                                            resolution=[30, 30, 30])
    cre = img.fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                                parameters={'density': 1.0e-7*u.cm**-3, 'spectral_index': -3.0})
    Bfield = img.fields.ConstantMagneticField(
        grid=cartesian_grid,
        parameters={'Bx': 6*u.microgauss,
                    'By': 0*u.microgauss,
                    'Bz': 0*u.microgauss})

    # Setup observing configuration
    observer = np.array([0, 0, 0])*u.kpc
    hIIdist = (box_size-2*u.kpc)*np.random.rand(Ndata) + 1*u.kpc  # uniform between [1, max-1] kpc
    dist_err = hIIdist/100000
    FB = get_label_FB(Ndata)
    config = {'grid': cartesian_grid,
              'observer': observer,
              'dist': hIIdist,
              'e_dist': dist_err,
              'lat': lat,
              'lon': lon,
              'FB': FB}

    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre, Bfield], mea=mea, config=config, noise=noise)
    sim_data = {'brightness': mock_data, 'err': mock_data*noise, 'lat': config['lat'], 'lon': config['lon']}
    sim_mea = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)

    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)

    # Setup field factories and their active parameters
    if fakemodel:  # overwrite Bfield with a different one from what was used for the simulated dataset
        rundir = 'runs/mockdatafake'
        Bfield = img.fields.ConstantMagneticField(
            grid=cartesian_grid,
            parameters={'Bx': 0*u.microgauss,
                        'By': 6*u.microgauss,
                        'Bz': 0*u.microgauss})
    else:
        rundir = 'runs/mockdata'

    B_factory = img.fields.FieldFactory(field_class=Bfield, grid=config['grid'])
    CRE_factory = img.fields.FieldFactory(field_class=cre, grid=config['grid'])
    CRE_factory.active_parameters = ('spectral_index',)
    CRE_factory.priors = {'spectral_index': img.priors.FlatPrior(xmin=-4, xmax=-2.1)}
    factory_list = [B_factory, CRE_factory]

    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline(simulator=los_simulator,
                                               run_directory=rundir,
                                               factory_list=factory_list,
                                               likelihood=likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}

    # Run!
    results = pipeline()

    return results


# pipeline controller
def JF12constindexCREprofile_setup(samplecondition=None):
    """
    Test retrieval of correct cre spectral index for regular JF12 magnetic field
    and an exponential number density CRE model with constant spectral index.

    Valid sample conditions are:
    - 'alpha'
    - 'allCRE'
    - 'Bamp+allCRE'

    """

    if samplecondition == None:
        print("Choose a valid sample condition!")
        return

    # Produce empty data format
    T = np.zeros(Ndata)*dunit  # placeholder
    T_err = np.zeros(Ndata)*dunit  # placeholder
    xmax = 15*u.kpc
    ymax = 15*u.kpc
    zmax = 2*u.kpc
    x = randrange(-0.9*xmax, 0.9*xmax, Ndata)
    y = randrange(-0.9*ymax, 0.9*ymax, Ndata)
    z = randrange(-0.9*zmax, 0.9*zmax, Ndata)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc, y, z)
    fake_data = {'brightness': T, 'err': T_err, 'lat': lat, 'lon': lon}
    mea = fill_imagine_dataset(data=fake_data)

    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                            resolution=[30, 30, 30])  # skipping x=y=0
    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                                parameters={'scale_radius': 10*u.kpc,
                                                            'scale_height': 1*u.kpc,
                                                            'central_density': 1e-5*u.cm**-3,
                                                            'spectral_index': -3})
    Bfield = WrappedJF12(grid=cartesian_grid)

    # Setup observing configuration
    observer = np.array([-8.5, 0, 0])*u.kpc
    dist_err = hIIdist/5
    FB = get_label_FB(Ndata)
    config = {'grid': cartesian_grid,
              'observer': observer,
              'dist': hIIdist,
              'e_dist': dist_err,
              'lat': lat,
              'lon': lon,
              'FB': FB}

    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre, Bfield], mea=mea, config=config, noise=0.01)
    sim_data = {'brightness': mock_data, 'err': mock_data/10, 'lat': config['lat'], 'lon': config['lon']}
    sim_mea = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)

    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)

    # Setup field factories and their active parameters
    B_factory = img.fields.FieldFactory(field_class=Bfield, grid=config['grid'])
    CRE_factory = img.fields.FieldFactory(field_class=cre, grid=config['grid'])
    if samplecondition == 'alpha':  # 1 parameter
        CRE_factory.active_parameters = ('spectral_index',)
        CRE_factory.priors = {'spectral_index': img.priors.FlatPrior(xmin=-4, xmax=-2.1)}
    if samplecondition == 'allCRE':  # 4 parameters
        CRE_factory.active_parameters = ('scale_radius', 'scale_height', 'central_density', 'spectral_index')
        CRE_factory.priors = {
            'scale_radius': img.priors.FlatPrior(xmin=5*u.kpc, xmax=15*u.kpc),
            'scale_height': img.priors.FlatPrior(xmin=0.1*u.kpc, xmax=2*u.kpc),
            'central_density': img.priors.FlatPrior(xmin=1e-6*u.cm**-3, xmax=1e-4*u.cm**-3),
            'spectral_index': img.priors.FlatPrior(xmin=-4, xmax=-2.1)}
    if samplecondition == 'Bamp+allCRE':  # 12 parameters
        CRE_factory.active_parameters = ('scale_radius', 'scale_height', 'central_density', 'spectral_index')
        CRE_factory.priors = {
            'scale_radius': img.priors.FlatPrior(xmin=5*u.kpc, xmax=15*u.kpc),
            'scale_height': img.priors.FlatPrior(xmin=0.1*u.kpc, xmax=2*u.kpc),
            'central_density': img.priors.FlatPrior(xmin=1e-6*u.cm**-3, xmax=1e-4*u.cm**-3),
            'spectral_index': img.priors.FlatPrior(xmin=-4, xmax=-2.1)}
        B_factory.active_parameters = ('b_arm_1', 'b_arm_2', 'b_arm_3', ' b_arm_4',
                                       'b_arm_5', 'b_arm_6', 'b_arm_7', 'b_ring')
        B_factory.priors = {
            'b_arm_1': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
            'b_arm_2': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
            'b_arm_3': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
            'b_arm_4': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
            'b_arm_5': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
            'b_arm_6': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
            'b_arm_7': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss),
            'b_ring': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss)}
    factory_list = [B_factory, CRE_factory]

    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline(simulator=los_simulator,
                                               run_directory=rundir,
                                               factory_list=factory_list,
                                               likelihood=likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}

    # Run!
    results = pipeline()

    summary = pipeline.posterior_summary
    samples = pipeline.samples

    return samples, summary


# pipeline controller
def turbulentJF12CREprofile_setup():

    # Produce empty data format
    T = np.zeros(Ndata)*dunit  # placeholder
    T_err = np.zeros(Ndata)*dunit  # placeholder
    xmax = 15*u.kpc
    ymax = 15*u.kpc
    zmax = 2*u.kpc
    x = randrange(-0.9*xmax, 0.9*xmax, Ndata)
    y = randrange(-0.9*ymax, 0.9*ymax, Ndata)
    z = randrange(-0.9*zmax, 0.9*zmax, Ndata)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc, y, z)
    fake_data = {'brightness': T, 'err': T_err, 'lat': lat, 'lon': lon}
    mea = fill_imagine_dataset(data=fake_data)

    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                            resolution=[30, 30, 30])  # skipping x=y=0
    # Setup observing configuration
    observer = np.array([-8.5, 0, 0])*u.kpc
    dist_err = hIIdist/100000
    FB = get_label_FB(Ndata)
    config = {'grid': cartesian_grid,
              'observer': observer,
              'dist': hIIdist,
              'e_dist': dist_err,
              'lat': lat,
              'lon': lon,
              'FB': FB}

    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                                parameters={'scale_radius': 10*u.kpc,
                                                            'scale_height': 1*u.kpc,
                                                            'central_density': 1e-5*u.cm**-3,
                                                            'spectral_index': -3})

    # Choose fixed JF12 field and possible active parameters
    Bfield1 = WrappedJF12(grid=cartesian_grid)
    B_factory = img.fields.FieldFactory(field_class=Bfield1, grid=cartesian_grid)
    B_factory.active_parameters = ('b_arm_1',)
    B_factory.priors = {'b_arm_1': img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss)}

    # Do sampling runs with different simulated datasets
    summary_t = []
    samples_t = []
    turbscale = [0.5]  # ,1.0,1.5]
    Barray = load_JF12rnd()
    for beta in turbscale:

        # Clear previous pipeline
        os.system("rm -r runs/mockdata/*")

        # Setup magnetic field
        Bfield2 = ArrayMagneticField(grid=cartesian_grid,
                                     array_field=Barray*u.microgauss,
                                     scale=beta,
                                     name='BrndJF12')
        Btotal = FieldAdder(grid=cartesian_grid, summand_1=Bfield1, summand_2=Bfield2)
        B_factory_total = img.fields.FieldFactory(field_class=Btotal, grid=cartesian_grid)

        # Produce simulated dataset with noise
        mock_data = produce_mock_data(field_list=[cre, Btotal], mea=mea, config=config, noise=0.01)
        sim_data = {'brightness': mock_data, 'err': mock_data/10, 'lat': config['lat'], 'lon': config['lon']}
        sim_mea = fill_imagine_dataset(sim_data)

        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)

        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)

        print(likelihood(mock_data))

        # Setup field factories and their active parameters
        CRE_factory = img.fields.FieldFactory(field_class=cre, grid=config['grid'])
        CRE_factory.active_parameters = ('spectral_index',)
        CRE_factory.priors = {'spectral_index': img.priors.FlatPrior(xmin=-4, xmax=-2.1)}
        factory_list = [B_factory, CRE_factory]

        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline(simulator=los_simulator,
                                                   run_directory=rundir,
                                                   factory_list=factory_list,
                                                   likelihood=likelihood)
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}

        # Run!
        results = pipeline()

        summary_t.append(pipeline.posterior_summary)
        samples_t.append(pipeline.samples)
    return turbscale, samples_t, summary_t
#results = turbulentJF12CREprofile_setup()

# pipeline controller


def JF12spectralhardeningCREprofile_setup():
    """
    Test retrieval of correct CRE hardening slope for a linear hardening model assuming
    the JF12 regular magnetic field.
    """

    # Produce empty data format
    T = np.zeros(Ndata)*dunit  # placeholder
    T_err = np.zeros(Ndata)*dunit  # placeholder
    xmax = 15*u.kpc
    ymax = 15*u.kpc
    zmax = 2*u.kpc
    x = randrange(-0.9*xmax, 0.9*xmax, Ndata)
    y = randrange(-0.9*ymax, 0.9*ymax, Ndata)
    z = randrange(-0.9*zmax, 0.9*zmax, Ndata)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc, y, z)
    fake_data = {'brightness': T, 'err': T_err, 'lat': lat, 'lon': lon}
    mea = fill_imagine_dataset(data=fake_data)

    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                            resolution=[30, 30, 30])  # skipping x=y=0
    cre_num = img.fields.CRENumberDensity(grid=cartesian_grid,
                                          parameters={'scale_radius': 10*u.kpc,
                                                      'scale_height': 1*u.kpc,
                                                      'central_density': 1e-5*u.cm**-3})
    cre_alpha = img.fields.SpectralIndexLinearVerticalProfile(
        grid=cartesian_grid,
        parameters={'soft_index': -4, 'hard_index': -2.5, 'slope': 1*u.kpc**-1})
    Bfield = WrappedJF12(grid=cartesian_grid)

    # Setup observing configuration
    observer = np.array([-8.5, 0, 0])*u.kpc
    dist_err = hIIdist/5
    FB = get_label_FB(Ndata)
    config = {'grid': cartesian_grid,
              'observer': observer,
              'dist': hIIdist,
              'e_dist': dist_err,
              'lat': lat,
              'lon': lon,
              'FB': FB}

    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre_num, cre_alpha, Bfield], mea=mea, config=config, noise=0.01)
    sim_data = {'brightness': mock_data, 'err': mock_data/10, 'lat': config['lat'], 'lon': config['lon']}
    sim_mea = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)

    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)

    # Setup field factories and their active parameters
    B_factory = img.fields.FieldFactory(field_class=Bfield, grid=config['grid'])
    cre_num_factory = img.fields.FieldFactory(field_class=cre_num, grid=config['grid'])
    alpha_factory = img.fields.FieldFactory(field_class=cre_alpha, grid=config['grid'])
    #alpha_factory.active_parameters=('slope',)
    #alpha_factory.priors = {'slope':img.priors.FlatPrior(xmin=0*u.kpc**-1, xmax=5*u.kpc**-1)}
    alpha_factory.active_parameters = (('slope', 'soft_index', 'hard_index'))
    alpha_factory.priors = {'slope': img.priors.FlatPrior(xmin=0*u.kpc**-1, xmax=5*u.kpc**-1),
                            'soft_index': img.priors.FlatPrior(xmin=-5, xmax=-3),
                            'hard_index': img.priors.FlatPrior(xmin=-3, xmax=-2.1)}
    factory_list = [cre_num_factory, alpha_factory, B_factory]

    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline(simulator=los_simulator,
                                               run_directory=rundir,
                                               factory_list=factory_list,
                                               likelihood=likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}

    # Run!
    results = pipeline()

    summary = pipeline.posterior_summary
    samples = pipeline.samples

    return samples, summary


#%% Choose which setup you want to run
#===================================================================================
#results = simple_pipeline()
#results = JF12constindexCREprofile_setup()
#results = JF12spectralhardeningCREprofile_setup()
#results = turbulentJF12CREprofile_setup()

#%% Results 3.1 Computational performance
#===================================================================================

def get_ndatavstime():
    """Save time testing results in binary np array"""
    datapoints = np.arange(10, 1010, 10)
    ctime = []
    for nd in datapoints:
        ctime.append(test_simulator(Ndata=nd, resolution=[30, 30, 30]))
    with open(logdir+'ndatavstime.npy', 'wb') as f:
        np.save(f, datapoints)
        np.save(f, np.array(ctime))
#get_ndatavstime()


def plot_ndatavstime():
    with open(logdir+'ndatavstime.npy', 'rb') as f:
        ndata = np.load(f)
        ctime = np.load(f)
    plt.plot(ndata, ctime)
    plt.ylim([0, 3])
    plt.title("Computationtime")
    plt.ylabel("time (s)")
    plt.xlabel("number of los")
    plt.savefig(figpath+'ndatavstime.png')
#plot_ndatavstime()


def get_resolutionvstime():
    res = [10, 20, 30, 40, 50, 60, 70]
    ctime = []  # total computation time of simulation
    itime = []  # initialization time of the simulator
    for r in res:
        tot, it = test_simulator(Ndata=50, resolution=[r, r, r], keepinit=True)
        ctime.append(tot)
        itime.append(it)
    with open(logdir+'resolutionvstime.npy', 'wb') as f:
        np.save(f, np.array(res))
        np.save(f, np.array(ctime))
        np.save(f, np.array(itime))
#get_resolutionvstime()


def plot_resolutionvstime():
    with open(logdir+'resolutionvstime.npy', 'rb') as f:
        res = np.load(f)
        ctime = np.load(f)
        itime = np.load(f)
    plt.plot(res, ctime, label='total')
    plt.plot(res, itime, label='initialization')
    plt.legend()
    plt.title("Computationtime")
    plt.ylabel("time (s)")
    plt.xlabel("grid resolution")
    plt.savefig(figpath+'resolutionvstime.png')
    plt.close('all')
    plt.plot(res, ctime-itime, label='total-init')
    plt.legend()
    plt.title("Computationtime")
    plt.ylabel("time (s)")
    plt.xlabel("grid resolution")
    plt.savefig(figpath+'resolutionvstime_relavent.png')
#plot_resolutionvstime()


def get_noisevsevidence():
    rel_brightness_error = 10**np.linspace(-2, 0, 20)
    reallogZ = []
    reallogZe = []
    fakelogZ = []
    fakelogZe = []
    for er in rel_brightness_error:
        os.system("rm -r runs/mockdata/*")
        os.system("rm -r runs/mockdatafake/*")
        real_results = simple_pipeline(noise=er, fakemodel=False)
        reallogZ.append(real_results['logZ'])
        reallogZe.append(real_results['logZerr'])
        fake_results = simple_pipeline(noise=er, fakemodel=True)
        fakelogZ.append(fake_results['logZ'])
        fakelogZe.append(fake_results['logZerr'])
    with open(logdir+'noisevsevidence.npy', 'wb') as f:
        np.save(f, np.array(rel_brightness_error))
        np.save(f, np.array(reallogZ))
        np.save(f, np.array(reallogZe))
        np.save(f, np.array(fakelogZ))
        np.save(f, np.array(fakelogZe))
#get_noisevsevidence()


def plot_noisevsevidence():
    with open(logdir+'noisevsevidence.npy', 'rb') as f:
        Te = np.load(f)
        rlZ = np.load(f)
        rlZe = np.load(f)
        flZ = np.load(f)
        flZe = np.load(f)
    plt.close('all')
    plt.plot(Te, rlZ, label='real model')
    #plt.fill_between(Te, rlZ-rlZe, rlZ+rlZe,color='gray', alpha=0.2) # errors on the evidence are tiny!!
    plt.plot(Te, flZ, label='fake model')
    #plt.fill_between(Te, flZ-flZe, flZ+flZe,color='gray', alpha=0.2)
    plt.xscale('log')
    plt.legend()
    plt.title("Brightness temperature noise performance")
    plt.ylabel("Evidence (logZ)")
    plt.xlabel("Relative brightness error Sigma_T")
    plt.savefig(figpath+'noisevsevidence.png')
#plot_noisevsevidence()


#%% Plotting routines for parameter inference runs
#===================================================================================


def plot_samples_seaborn(samp, colnames, fname):

    def show_truth_in_jointplot(jointplot, true_x, true_y, color='r'):
        for ax in (jointplot.ax_joint, jointplot.ax_marg_x):
            ax.vlines([true_x], *ax.get_ylim(), colors=color)
        for ax in (jointplot.ax_joint, jointplot.ax_marg_y):
            ax.hlines([true_y], *ax.get_xlim(), colors=color)

    snsfig = sns.jointplot(data=samp, kind='kde')
    snsfig.plot_joint(sns.scatterplot, linewidth=0, marker='.', color='0.3')
    #show_truth_in_jointplot(snsfig, a0, b0)
    plt.savefig(fname)


def plot_seaborn_corner(samples, colnames, fname):
    #print(samples)
    df = pd.DataFrame(data=samples, columns=colnames)
    fig = sns.pairplot(data=df, corner=True, kind='kde')
    plt.savefig(fname)


#%% Results 4.2 Sampeling parameters of JF12 + Const-Index CRE profile
#=======================================================================================

# ======= Just retrieve spectral index =======
def get_samples_alpha():
    os.system("rm -r runs/mockdata/*")
    samples, summary = JF12constindexCREprofile_setup(samplecondition='alpha')
    with open(logdir+'samples_alpha.npy', 'wb') as f:
        np.save(f, summary)
        np.save(f, samples)
#get_samples_alpha()


def plot_samples_alpha():
    with open(logdir+'samples_alpha.npy', 'rb') as f:
        summary = np.load(f, allow_pickle=True)
        samples = np.load(f)
    npsamp = []
    for s in samples:
        npsamp.append(s[0])
    plt.hist(npsamp)
    plt.savefig(figpath+'samplesalpha.png')
#plot_samples_alpha()


# ======= Retrieve all CRE model parameters at once =======
def get_samples_CRE():
    os.system("rm -r runs/mockdata/*")
    samples, summary = JF12constindexCREprofile_setup(samplecondition='allCRE')
    with open(logdir+'samples_CRE.npy', 'wb') as f:
        np.save(f, summary)
        np.save(f, samples)
#get_samples_CRE()


def plot_samples_CRE():
    with open(logdir+'samples_CRE.npy', 'rb') as f:
        summary = np.load(f, allow_pickle=True)
        samples = np.load(f)
    npsamp = []
    for s in samples:
        npsamp.append(list(s))
    npsamp = np.array(npsamp)
    names = ('scale_radius', 'scale_height', 'central_density', 'spectral_index')
    plt.close('all')
    plot_seaborn_corner(samples=npsamp, colnames=names, fname=figpath+'samplesCRE.png')
#plot_samples_CRE()


# ======= Retrieve all B amplitudes for JF12 and all CRE model parameters at once =======
def get_samples_JF12andCRE():
    os.system("rm -r runs/mockdata/*")
    samples, summary = JF12constindexCREprofile_setup(samplecondition='Bamp+allCRE')
    with open(logdir+'samples_JF12CRE.npy', 'wb') as f:
        np.save(f, summary)
        np.save(f, samples)
#get_samples_CRE()


def plot_samples_JF12andCRE():
    with open(logdir+'samples_JF12CRE.npy', 'rb') as f:
        summary = np.load(f, allow_pickle=True)
        samples = np.load(f)
    npsamp = []
    for s in samples:
        npsamp.append(list(s))
    npsamp = np.array(npsamp)
    names = ('b_arm_1', 'b_arm_2', 'b_arm_3', ' b_arm_4', 'b_arm_5', 'b_arm_6', 'b_arm_7', 'b_ring')
    names += ('scale_radius', 'scale_height', 'central_density', 'spectral_index')
    plt.close('all')
    plot_seaborn_corner(samples=npsamp, colnames=names, fname=figpath+'samplesCRE.png')
#plot_samples_JF12andCRE()


# ======= Investigate turbulent setup =======
def get_samples_turbulence():
    os.system("rm -r runs/mockdata/*")
    beta, samples_t, summary_t = turbulentJF12CREprofile_setup()
    with open(logdir+'samples_turbulence.npy', 'wb') as f:
        np.save(f, summary)
        np.save(f, samples)
#get_samples_turbulence()


def plot_samples_turbulence():
    with open(logdir+'samples_turbulence.npy', 'rb') as f:
        summary_t = np.load(f, allow_pickle=True)
        samples_t = np.load(f)
    # process data
    # make several line+scatter plots for samples of parameter_x vs beta
    # line = b0 + beta*b0_rand
    # samples = samples column x for each beta
#plot_samples_turbulence()
