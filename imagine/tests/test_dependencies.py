# %% IMPORTS
# Package imports
import astropy.units as u
import numpy as np
import pytest

# IMAGINE imports
from imagine.fields import (
    DummyField, MagneticField, ThermalElectronDensityField, UniformGrid)
from imagine.observables import Measurements, TabularDataset
from imagine.simulators import Simulator

# Marks tests in this module as quick
pytestmark = pytest.mark.quick

# %% HELPER DEFINITIONS

# First, we create some fields with the structure:
# A - independent - electron density
# B - independent - electron density
# C -> B - dummy
# D -> (C, B) - magnetic field
# E -> electron density (A,B) - magnetic field
# F - independent - magnetic field


class A(ThermalElectronDensityField):
    """Independent electron density"""

    NAME = 'A'
    PARAMETER_NAMES = []

    def compute_field(self, seed):
        return np.ones(self.data_shape)*self.units


class B(ThermalElectronDensityField):
    """Independent electron density"""

    NAME = 'B'
    PARAMETER_NAMES = []

    def compute_field(self, seed):
        self.secret = 9  # Example of shared information
        return np.ones(self.data_shape)*self.units/2.


class C(DummyField):
    """Dummy field dependent on B"""

    NAME = 'C'
    FIELD_CHECKLIST = {}
    SIMULATOR_CONTROLLIST = {}
    DEPENDENCIES_LIST = [B]


class D(MagneticField):
    """
    Magnetic field, dependent on B and C

    Each component takes the secret number saved during the evaluation of B
    """

    NAME = 'D'
    PARAMETER_NAMES = []
    DEPENDENCIES_LIST = [B, C]

    def compute_field(self, seed):
        result = np.ones(self.data_shape)*self.units

        return self.dependencies[B].secret * result


class E(MagneticField):
    """
    Magnetic field, dependent total thermal electron density

    Each component takes the numerical value of the electron density
    """

    NAME = 'E'
    PARAMETER_NAMES = []
    DEPENDENCIES_LIST = ['thermal_electron_density']

    def compute_field(self, seed):
        te_density = self.dependencies['thermal_electron_density']
        B = np.empty(self.data_shape)
        for i in range(3):
            B[..., i] = te_density.value
        return B*u.microgauss


class F(MagneticField):
    """Independent magnetic field"""

    NAME = 'F'
    DEPENDENCIES_LIST = [B, C]
    PARAMETER_NAMES = []

    def compute_field(self, seed):
        return np.ones(self.data_shape)*0.1*u.microgauss


# We initalize a common grid for all the tests
grid = UniformGrid([[0, 1]]*3*u.kpc, resolution=[1]*3)


class DummySimulator(Simulator):
    # Class attributes
    SIMULATED_QUANTITIES = ['nothing']
    REQUIRED_FIELD_TYPES = ['dummy', 'magnetic_field',
                            'thermal_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']

    def simulate(self, key, coords_dict, realization_id, output_units):

        results = np.zeros(coords_dict['lat'].size)

        print('The supplied (total) fields are:\n')
        for name in self.fields:
            print(name)
            print('\t', self.fields[name])
            if name != 'dummy':
                results += self.fields[name].value.sum()
        print()

        return results*output_units


# %% PYTEST DEFINITIONS
def test_Field_dependency():
    # Initializes the fields
    a = A(grid)
    b = B(grid)
    c = C(grid)
    d = D(grid)
    e = E(grid)

    # Checks whether dependencies on classes are working
    b.get_data()  # Needs to evaluate these once
    c.get_data(dependencies={B: b})  # Needs to evaluate these once
    result = d.get_data(dependencies={B: b, C: c})
    assert np.all(result == [[[9]*3]]*u.microgauss)

    # Checks whether dependencies on types are working
    te_density = a.get_data()+b.get_data()
    result = e.get_data(dependencies={'thermal_electron_density': te_density})
    assert np.all(result == [[[1.5]*3]]*u.microgauss)


def test_Simulator_dependency_resolution():
    dat = TabularDataset({'data': [0], 'lat': 0, 'lon': 0, 'err': 0.1},
                         name='nothing',
                         units=u.rad,
                         data_col='data',
                         err_col='err')
    mea = Measurements()
    mea.append(dataset=dat)

    sim = DummySimulator(mea)

    fields_list = list(map(lambda x: x(grid), (F, E, D, C, B, A)))
    obs = sim(fields_list)

    assert obs[('nothing', None, 'tab', None)].global_data[0][0] == 33.3
