import numpy as np

from ..TOGOBaseModels import MagneticField


class ConstantMagneticField(MagneticField):
    """
    Constant magnetic field

    The field parameters are:
    'Bx', 'By', 'Bz', which correspond to the fixed components
    :math:`B_x`, :math:`B_x` and :math:`B_z`.
    """

    def __init__(self, grid):

        super().__init__(self, grid, ['Bx', 'By', 'Bz'], call_by_subclass=True)


    def compute_field(self, parameters):
        # Creates an empty array to store the result
        B = np.empty(self.data_shape) * parameters['Bx']*self.unit
        # For a magnetic field, the output must be of shape:
        # (Nx,Ny,Nz,Nc) where Nc is the index of the component.
        # Computes Bx
        B[:, :, :, 0] = parameters['Bx']
        # Computes By
        B[:, :, :, 1] = parameters['By']
        # Computes Bz
        B[:, :, :, 2] = parameters['Bz']
        return B
