"""
This module contains a set of pytest-compatible test functions which check
the interfaces exemplified in the templates, supplemented by the functions
and classes defined in the mock_for_templates module
"""
# %% IMPORTS
# Package imports
import astropy.units as u
import numpy as np
import pytest
import os

# IMAGINE imports
import imagine.tests.mocks_for_templates as mock
import imagine.fields as img_fields
import imagine.priors as img_priors
import imagine.observables as img_obs
from imagine.likelihoods import SimpleLikelihood
from imagine.simulators import TestSimulator
from imagine import load_pipeline
from imagine import rc

# imports templates
from imagine.templates.magnetic_field_template import MagneticFieldTemplate
from imagine.templates.thermal_electrons_template import ThermalElectronsDensityTemplate
from imagine.templates.field_factory_template import FieldFactoryTemplate
from imagine.templates.simulator_template import SimulatorTemplate
from imagine.templates.pipeline_template import PipelineTemplate


__all__ = []

# Marks tests in this module as quick
pytestmark = pytest.mark.quick

# Convenience
muG = u.microgauss

# %% PYTEST DEFINITIONS
def test_magnetic_field_template():
    """
    Tests the MagneticFieldTemplate, including the handling of cartesian units
    and units by Fields.
    """
    grid = img_fields.UniformGrid(box=[[-1*u.kpc, 1*u.kpc]]*3,
                                       resolution=[2]*3)

    magnetic_field = MagneticFieldTemplate(grid,
                                           parameters={'Parameter_A': 17*muG,
                                                      'Parameter_B': 2})
    data = magnetic_field.get_data()
    Bx = data[:,:,:,0]
    By = data[:,:,:,1]
    Bz = data[:,:,:,2]

    assert np.all(Bx == 17*muG)
    assert np.all(By == 2*muG)
    assert Bz[0,0,0] == -42*muG
    assert Bz[1,1,1] == 42*muG


def test_thermal_electrons_template():
    """
    Tests the ThermalElectronsDensityTemplate, including the handling of
    spherical coordinates and ensemble seeds.
    """
    grid = img_fields.UniformGrid(box=[[1,2]*u.kpc, [1,2]*u.rad, [1,2]*u.rad],
                                  resolution=[2]*3, grid_type='spherical')

    ne = ThermalElectronsDensityTemplate(grid, ensemble_seeds=[1,2],
                                         parameters={'Parameter_A': 2000*u.pc,
                                                     'Parameter_B': 0.5})
    # The answer combines the spherical coordinate values and the 2 parameters
    answer = [[[1., 2.], [2., 4.]], [[2., 4.], [4., 8.]]] << u.cm**-3

    assert(np.allclose(ne.get_data(0).cgs, answer))
    assert(np.allclose(ne.get_data(1).cgs, answer*2))


def test_field_factory_template():
    """
    Tests the FieldFactoryTemplate

    Checks whether FieldFactory object is initialized,
    providing one inactive/default parameter and one active parameter.

    Asks the factory to produce a "sample" of the Field, using the Prior.
    """
    grid = img_fields.UniformGrid(box=[[-1*u.kpc, 1*u.kpc]]*3,
                                  resolution=[2]*3)
    field_factory = FieldFactoryTemplate(grid=grid,
                                         active_parameters=['Parameter_B'])

    field = field_factory(variables={'Parameter_B': 0.65})

    assert isinstance(field, mock.MY_PACKAGE.MY_FIELD_CLASS)
    assert field.parameters['Parameter_A'] == 1*u.K
    print(field.parameters['Parameter_B'])
    assert np.isclose(field.parameters['Parameter_B'], 0.65)


def test_simulator_template():
    """
    Tests the SimulatorTemplate
    """
    measurements = img_obs.Measurements()
    fake = {'dat': [0,1], 'err':[0,0],
            'lat': [-1,0]*u.deg, 'lon': [2,3]*u.rad}
    dset = img_obs.TabularDataset(fake, name='my_observable_quantity',
                                  tag='I', frequency=20*u.cm, units=u.jansky,
                                  data_col='dat', err_col='err')
    measurements.append(dataset=dset)

    simulator = SimulatorTemplate(measurements)

    grid = img_fields.UniformGrid(box=[[0*u.kpc, 1*u.kpc]]*3, resolution=[2]*3)
    B = img_fields.ConstantMagneticField(grid,
                                         parameters={'Bx': 42*muG,
                                         'By':  1*muG,
                                         'Bz':  0*muG})
    ne = img_fields.ConstantThermalElectrons(grid,
                                             parameters={'ne': 1000*u.m**-3})
    dummy = mock.MockDummy(parameters={'value': -100000, 'units': 1*u.jansky})

    simulations = simulator([B, ne, dummy])
    obs = simulations[dset.key]
    assert np.allclose(obs.global_data, [[25.48235893, 42.]])



class ConstantBFactory(img_fields.FieldFactory):
    """Example: field factory for YourFieldClass"""

    # Class attributes
    # Field class this factory uses
    FIELD_CLASS = img_fields.ConstantMagneticField

    # Default values are used for inactive parameters
    DEFAULT_PARAMETERS = {'Bx': 1.*muG, 'By': 2.*muG, 'Bz': 3.*muG}

    # All parameters need a range and a prior
    # this tests: FlatPrior, GaussianPrior, GaussianPrior (truncated)
    PRIORS = {'Bx': img_priors.GaussianPrior(mu=1.5*muG, sigma=0.5*muG,
                                             xmin=0*muG, xmax=5.0*muG),
              'By': img_priors.GaussianPrior(mu=1.5*muG, sigma=0.5*muG),
              'Bz': img_priors.FlatPrior(xmin=0*muG, xmax=1.*muG)}


class FakeRandomTE(img_fields.ThermalElectronDensityField):
    # Class attributes
    NAME = 'fake_rnd_TE'

    STOCHASTIC_FIELD = True
    DEPENDENCIES_LIST = []
    PARAMETER_NAMES = ['param']

    def compute_field(self, seed):
        # One can access the parameters supplied in the following way
        return np.ones(self.data_shape)*u.cm**(-3)

class FakeRandomTEFactory(img_fields.FieldFactory):
    """Example: field factory for YourFieldClass"""
    FIELD_CLASS = FakeRandomTE
    DEFAULT_PARAMETERS = {'param':2}
    PRIORS = {'param': img_priors.FlatPrior(xmin=0, xmax=10.)}


def test_pipeline_template():
    """
    Tests the PipelineTemplate
    """
    # Fake measurements / covariances
    fd_units = u.microgauss*u.cm**-3
    x = np.arange(5)
    fd = np.ones_like(x)
    data = {'meas' : fd,
            'err': np.ones_like(fd)*0.1,
            'x': x,
            'y': np.zeros_like(fd),
            'z': np.zeros_like(fd)}
    dset = img_obs.TabularDataset(data, name='test',
                                  data_col='meas',
                                  coords_type='cartesian',
                                  x_col='x', y_col='y',
                                  z_col='z', err_col='err',
                                  units=fd_units)
    measurements = img_obs.Measurements()
    covariances = img_obs.Covariances()
    measurements.append(dataset=dset)
    covariances.append(dataset=dset)

    # Likelihood
    likelihood = SimpleLikelihood(measurements, covariances)

    # Grid
    grid = img_fields.UniformGrid(box=[[0,2*np.pi]*u.kpc,
                                       [0,0]*u.kpc,
                                       [0,0]*u.kpc],
                                  resolution=[30,1,1])
    # Field factories
    TE_factory = FakeRandomTEFactory(grid=grid)
    TE_factory.active_parameters = ['param']
    B_factory = ConstantBFactory(grid=grid)
    B_factory.active_parameters = ['Bx', 'By']

    # Simulator
    simulator = TestSimulator(measurements)

    # Sets the pipeline
    run_directory = os.path.join(rc['temp_dir'], 'test_templates')
    pipeline = PipelineTemplate(run_directory=run_directory,
                                simulator=simulator,
                                factory_list=[TE_factory, B_factory],
                                likelihood=likelihood,
                                ensemble_size=2)
    # Tests sampling controlers
    pipeline.sampling_controllers = dict(controller_a=True)

    # Runs fake pipeline, including another sampling controller
    # This in turn checks multiple structures of the pipeline object
    pipeline(controller_b=False)

    # Tests posterior report (checks execution only)
    pipeline.posterior_report()
    # Tests posterior summary
    assert pipeline.posterior_summary['constant_B_Bx']['median']==0.5*muG
    assert pipeline.posterior_summary['constant_B_By']['median']==0.5*muG

    # Tests (temporary) chains and run directory creation
    run_dir = pipeline.run_directory
    assert os.path.isdir(pipeline.chains_directory)
    assert os.path.isdir(run_dir)
    # checks ("computed") log_evidence
    assert (pipeline.log_evidence, pipeline.log_evidence_err) == (42.0, 17.0)

    # Tests saving and loading
    # (the pipeline should have been saved after )
    pipeline_copy = load_pipeline(pipeline.run_directory)
    assert (pipeline_copy.log_evidence,
            pipeline_copy.log_evidence_err) == (42.0, 17.0)
    assert pipeline_copy.posterior_summary['constant_B_By']['median']==0.5*muG
