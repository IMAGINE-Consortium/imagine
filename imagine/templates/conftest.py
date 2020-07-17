# Settings for the test_templates test_templates
# Ignore this file unless you know what you are doing!
#
# Here we set up several mock modules which allow testing the template files
#
import sys
import numpy as np
import astropy.units as u
from imagine.fields import DummyField

# For testing the magnetic_field_template
def MY_GMF_MODEL_compute_B(param_A, param_B, x_coord, y_coord, z_coord, seed):
    # Checks interface with physical parameters
    Bx = param_A
    # Checks interface with dimensionless parameters
    By = param_B * 1e-10 *u.tesla # This will also test unit conversions..
    # Checks cartesian coordinates
    Bz = 42*(x_coord.value*y_coord.value*z_coord.value) * u.microgauss
    return Bx, By, Bz

module = type(sys)('MY_GMF_MODEL')
module.compute_B = MY_GMF_MODEL_compute_B
sys.modules['MY_GMF_MODEL'] = module

# For testing the thermal_electrons_template
def MY_GALAXY_MODEL_compute_ne(param_A, param_B, r, theta, phi, seed):
    # Checks interface with physical parameters
    A = param_A / (1*u.kpc)
    # Checks interface with dimensionless parameters
    B = param_B * 1*u.m**-3
    # Checks spherical coordinates
    return A*B*(r.value*theta.value*phi.value)

module = type(sys)('MY_GALAXY_MODEL')
module.compute_ne = MY_GALAXY_MODEL_compute_ne
sys.modules['MY_GALAXY_MODEL'] = module

# For testing the field_factory_template
class MY_PACKAGE_MY_FIELD_CLASS(DummyField):

    # Class attributes
    NAME = 'name_of_the_dummy_field'

    @property
    def field_checklist(self):
        return {'Parameter_A': 'parameter_A_settings',
                'Parameter_B': None}

module = type(sys)('MY_PACKAGE')
module.MY_FIELD_CLASS = MY_PACKAGE_MY_FIELD_CLASS
sys.modules['MY_PACKAGE'] = module

# For testing the simulator_template
def MY_SIMULATOR_simulate():
    pass

module = type(sys)('MY_SIMULATOR')
module.simulate = MY_SIMULATOR_simulate
sys.modules['MY_SIMULATOR'] = module

# For testing the pipeline_template
class MY_SAMPLER_Sampler:
    def __init__(param_names=None, loglike=None, prior_transform=None,
                 prior_pdf=None):
        pass

    def run(**kwargs):
        return {'samples': [0.5,0.5], 'logz': 42, 'logzerr': 17}

module = type(sys)('MY_SAMPLER')
module.Sampler = MY_SAMPLER_Sampler
sys.modules['MY_SAMPLER'] = module




