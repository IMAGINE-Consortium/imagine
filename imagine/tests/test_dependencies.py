import imagine as img
import numpy as np
import astropy.units as u

# First, we create some fields with the structure:
# A - independent - electron density
# B - independent - electron density
# C -> B - dummy
# D -> (C, B) - magnetic field
# E -> electron density (A,B) - magnetic field
# F - independent - magnetic field

class A(img.ThermalElectronDensityField):
    """Independent electron density"""
    field_name = 'A'
    stochastic_field = False

    def compute_field(self, seed):
        return np.ones(self.data_shape)*self.field_units

class B(img.ThermalElectronDensityField):
    """Independent electron density"""
    field_name = 'B'
    stochastic_field = False

    def compute_field(self, seed):
        self.secret = 9 # Example of shared information
        return np.ones(self.data_shape)*self.field_units/2.

class C(img.DummyField):
    """Dummy field dependent on B"""
    field_name = 'C'
    stochastic_field = False

    @property
    def dependencies_list(self):
        return [B]

class D(img.MagneticField):
    """
    Magnetic field, dependent on B and C

    Each component takes the secret number saved during the evaluation of B
    """
    field_name = 'D'
    stochastic_field = False

    @property
    def dependencies_list(self):
        return [C,B]
    def compute_field(self, seed):
        result = np.ones(self.data_shape)*self.field_units

        return self.dependencies[B].secret * result

class E(img.MagneticField):
    """
    Magnetic field, dependent total thermal electron density

    Each component takes the numerical value of the electron density
    """
    field_name = 'E'
    stochastic_field = False

    @property
    def dependencies_list(self):
        return ['thermal_electron_density']

    def compute_field(self, seed):
        te_density = self.dependencies['thermal_electron_density']
        B = np.empty(self.data_shape)
        for i in range(3):
            B[...,i] = te_density.value
        return B*u.microgauss

class F(img.MagneticField):
    """Independent magnetic field"""
    field_name = 'F'
    stochastic_field = False

    def compute_field(self, seed):
        return np.ones(self.data_shape)*0.1*u.microgauss

# We initalize a common grid for all the tests
grid = img.UniformGrid([[0,1]]*3*u.kpc,resolution=[1]*3)

class DummySimulator(img.Simulator):

    @property
    def simulated_quantities(self):
        return {'nothing'}
    @property
    def required_field_types(self):
        return {'dummy','magnetic_field', 'thermal_electron_density'}
    @property
    def allowed_grid_types(self):
        return {'cartesian'}

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


def test_Field_dependency():
    # Initializes the fields
    a = A(grid)
    b = B(grid)
    c = C(grid)
    d = D(grid)
    e = E(grid)

    # Checks whether dependencies on classes are working
    b.get_data()  # Needs to evaluate these once
    c.get_data(dependencies={B:b}) # Needs to evaluate these once
    result = d.get_data(dependencies={B:b, C:c})
    assert np.all(result == [[[9]*3]]*u.microgauss)

    # Checks whether dependencies on types are working
    te_density = a.get_data()+b.get_data()
    result = e.get_data(dependencies={'thermal_electron_density': te_density})
    assert np.all(result == [[[1.5]*3]]*u.microgauss)


def test_Simulator_dependency_resolution():
    dat = img.observables.TabularDataset({'x':[0],'lat':0,'lon':0,'err':0.1},
                                         name='nothing',
                                         units=u.rad,
                                         data_column='x',
                                         error_column='err',
                                         lat_column='lat',
                                         lon_column='lon')
    mea = img.Measurements()
    mea.append(dataset=dat)

    sim = DummySimulator(mea)

    fields_list = [field(grid) for field in (F,E,D,C,B,A)]
    obs = sim(fields_list)

    assert obs[('nothing', 'nan', 'tab', 'nan')].global_data[0][0] == 33.3