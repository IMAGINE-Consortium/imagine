"""
For testing purposes only
"""


# %% IMPORTS
# Package imports
import astropy.units as u
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# IMAGINE imports
from imagine.simulators import Simulator

# All declaration
__all__ = ['TestSimulator']


# %% CLASS DEFINITIONS
class TestSimulator(Simulator):
    r"""
    Example simulator for illustration and testing

    Computes a Faraday-depth-like property at a given point without
    performing the integration, i.e. computes:

    .. math ::
        $t(x,y,z) = B_y\,n_e\,$

    """

    # Class attributes
    SIMULATED_QUANTITIES = ['test']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'thermal_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']

    def __init__(self, measurements, LoS_axis='y'):
        # Send the measurenents to parent class
        super().__init__(measurements)
        if LoS_axis=='y':
            self.B_axis = 1
        elif LoS_axis=='z':
            self.B_axis = 2
        else:
            raise ValueError

    def simulate(self, key, coords_dict, realization_id, output_units):
        # Accesses fields and grid

        Bpara = self.fields['magnetic_field'][:,:,:,self.B_axis]
        ne = self.fields['thermal_electron_density']
        x = self.grid.x[:,0,0].to_value(u.kpc)
        y = self.grid.y[0,:,0].to_value(u.kpc)
        z = self.grid.z[0,0,:].to_value(u.kpc)

        fd = (Bpara*ne).to_value(output_units)

        # Converts the grids to a format compatible with the interpolator
        # (comment: this is a bit silly, but what is the native numpy alternative?)
        fd_interp = RegularGridInterpolator(points=(x, y, z),
                                            values=fd, method='nearest')

        interp_points = np.array([coords_dict[c].to_value(u.kpc)
                                  for c in ('x', 'y', 'z')]).T

        with np.errstate(invalid='ignore', divide='ignore'):
            results = fd_interp(interp_points)*output_units

        return results
