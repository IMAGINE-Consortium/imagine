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

# IMAGINE imports
import imagine.tests.mocks_for_templates as mock
from imagine.fields import UniformGrid
from imagine.templates.magnetic_field_template import MagneticFieldTemplate
from imagine.templates.thermal_electrons_template import ThermalElectronsDensityTemplate
from imagine.templates.field_factory_template import FieldFactoryTemplate
from imagine.templates.simulator_template import SimulatorTemplate
from imagine.fields import ConstantMagneticField, ConstantThermalElectrons
from imagine.fields import DummyField
from imagine.observables import TabularDataset, Measurements

__all__ = []

# Marks tests in this module as quick
pytestmark = pytest.mark.quick

# %% PYTEST DEFINITIONS
def test_magnetic_field_template():
    """
    Tests the MagneticFieldTemplate, including the handling of cartesian units
    and units by Fields.
    """
    grid = UniformGrid(box=[[-1*u.kpc, 1*u.kpc]]*3,
                                       resolution=[2]*3)

    magnetic_field = MagneticFieldTemplate(grid,
                                           parameters={'Parameter_A': 17*u.microgauss,
                                                      'Parameter_B': 2})
    data = magnetic_field.get_data()
    Bx = data[:,:,:,0]
    By = data[:,:,:,1]
    Bz = data[:,:,:,2]

    assert np.all(Bx == 17*u.microgauss)
    assert np.all(By == 2*u.microgauss)
    assert Bz[0,0,0] == -42*u.microgauss
    assert Bz[1,1,1] == 42*u.microgauss


def test_thermal_electrons_template():
    """
    Tests the ThermalElectronsDensityTemplate, including the handling of
    spherical coordinates and ensemble seeds.
    """
    grid = UniformGrid(box=[[1,2]*u.kpc, [1,2]*u.rad, [1,2]*u.rad],
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
    grid = UniformGrid(box=[[-1*u.kpc, 1*u.kpc]]*3,
                                       resolution=[2]*3)
    field_factory = FieldFactoryTemplate(grid=grid,
                                         active_parameters=['Parameter_B'])

    field = field_factory(variables={'Parameter_B': 0.65})

    assert isinstance(field, mock.MY_PACKAGE.MY_FIELD_CLASS)
    assert field.parameters['Parameter_A'] == 1*u.K
    assert np.isclose(field.parameters['Parameter_B'], 1.3*u.Msun)


def test_simulator_template():
    """
    Tests the SimulatorTemplate
    """
    measurements = Measurements()
    fake = {'dat': [0,1], 'err':[0,0],
            'lat': [-1,0]*u.deg, 'lon': [2,3]*u.rad}
    dset = TabularDataset(fake, name='my_observable_quantity',
                          tag='I', frequency=20*u.cm, units=u.jansky,
                          data_column='dat', error_column='err',
                          lat_column='lat', lon_column='lon')
    measurements.append(dataset=dset)

    simulator = SimulatorTemplate(measurements)

    grid = UniformGrid(box=[[0*u.kpc, 1*u.kpc]]*3, resolution=[2]*3)
    B = ConstantMagneticField(grid, parameters={'Bx': 42*u.microgauss,
                                                'By':  1*u.microgauss,
                                                'Bz':  0*u.microgauss})
    ne = ConstantThermalElectrons(grid, parameters={'ne': 1000*u.m**-3})
    dummy = mock.MockDummy(parameters={'value': -100000, 'units': 1*u.jansky})

    simulations = simulator([B, ne, dummy])
    obs = simulations[dset.key]
    assert np.allclose(obs.global_data, [[25.48235893, 42.]])
