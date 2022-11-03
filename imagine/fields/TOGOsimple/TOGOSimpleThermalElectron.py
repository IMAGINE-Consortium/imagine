import numpy as np

from ..TOGOBaseModels import ThermalElectronDensityField


class ConstantThermalElectrons(ThermalElectronDensityField):
    """
    Constant thermal electron density field

    The field parameters are:
    'ne', the number density of thermal electrons
    """

    def compute_field(self, seed):
        return np.ones(self.data_shape)*self.parameters['ne']*self.unit


class ExponentialThermalElectrons(ThermalElectronDensityField):
    """
    Thermal electron distribution in a double exponential disc
    characterized by a scale-height and a scale-radius, i.e.

    ..math::

        n_e(R) = n_0 e^{-R/R_e} e^{-|z|/h_e}

    where :math:`R` is the cylindrical radius and :math:`z` is the vertical
    coordinate.

    The field parameters are: the 'central_density', `n_0`;
    'scale_radius`, :math:`R_e`; and 'scale_height', :math:`h_e`.
    """

    # Class attributes
    NAME = 'exponential_disc_thermal_electrons'
    PARAMETER_NAMES = ['central_density',
                       'scale_radius',
                       'scale_height']

    def compute_field(self, seed):
        R = self.grid.r_cylindrical
        z = self.grid.z
        Re = self.parameters['scale_radius']
        he = self.parameters['scale_height']
        n0 = self.parameters['central_density']

        return n0*np.exp(-R/Re)*np.exp(-np.abs(z/he))
