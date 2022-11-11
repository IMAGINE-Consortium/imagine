import numpy as np
import astropy.units as apu

from ..TOGOBaseModels import VectorField


class ConstantMagneticField(VectorField):
    """
    Constant magnetic field

    The field parameters are:
    'Bx', 'By', 'Bz', which correspond to the fixed components
    :math:`B_x`, :math:`B_x` and :math:`B_z`.
    """

    def __init__(self, grid):
        parameter_def = {'Bx': apu.microgauss, 'By': apu.microgauss, 'Bz': apu.microgauss}
        super().__init__(grid, parameter_def, apu.microgauss, 3, call_by_method=True)

    def compute_model(self, parameters):
        # Creates an empty array to store the result
        B = np.empty(self.data_shape) * parameters['Bx']
        # For a magnetic field, the output must be of shape:
        # (Nx,Ny,Nz,Nc) where Nc is the index of the component.
        # Computes Bx
        B[:, :, :, 0] = parameters['Bx']
        # Computes By
        B[:, :, :, 1] = parameters['By']
        # Computes Bz
        B[:, :, :, 2] = parameters['Bz']
        return B
