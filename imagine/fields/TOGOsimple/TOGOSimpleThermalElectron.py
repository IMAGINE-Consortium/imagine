import numpy as np

from ..TOGOBaseModels import ThermalElectronDensityField


class ConstantThermalElectron(ThermalElectronDensityField):
    """
    Constant thermal electron density field

    The field parameters are:
    'ne', the number density of thermal electrons
    """

    def __init__(self, grid):

        super().__init__(grid, ['ne'], call_by_method=True)

    def compute_model(self, parameters):
        return np.ones(self.data_shape)*parameters['ne']


class ExponentialThermalElectron(ThermalElectronDensityField):
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

    def __init__(self, grid):
        super().__init__(grid, ['central_density', 'scale_radius', 'scale_height'], call_by_method=True)

    def compute_model(self, parameters):
        R = self.grid.r_cylindrical
        z = self.grid.z
        Re = parameters['scale_radius']
        he = parameters['scale_height']
        n0 = parameters['central_density']

        return n0*np.exp(-R/Re)*np.exp(-np.abs(z/he))
