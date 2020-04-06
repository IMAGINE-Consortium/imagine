"""
built only for testing purposes only
"""

import numpy as np
import astropy.units as u
from imagine import Simulator
from imagine.tools.icy_decorator import icy
from scipy.interpolate import RegularGridInterpolator

@icy
class TestSimulator(Simulator):
    """ 
    Example simulator for illustration and testing
    
    Computes a Faraday-depth-like property at a given point without
    performing the integration, i.e. computes:
    
    .. math ::
        B
        
    
    """
    def __init__(self, measurements, LoS_axis='y'):
        # Send the measurenents to parent class
        super().__init__(measurements)
        if LoS_axis=='y':
            self.B_axis = 1
        elif LoS_axis=='z':
            self.B_axis = 2
        else:
            raise valueError
            
    @property
    def simulated_quantities(self):
        return {'test'}
    @property
    def required_field_types(self):
        return {'magnetic_field', 'thermal_electron_density'}
    @property
    def allowed_grid_types(self):
        return {'cartesian'}
    
    def simulate(self, key, coords_dict, Nside, output_units):
        # Accesses fields and grid 
        
        Bpara = self.fields['magnetic_field'][:,:,:,self.B_axis]
        ne = self.fields['thermal_electron_density']
        x = self.grid.x.to_value(u.kpc)
        y = self.grid.y.to_value(u.kpc)
        z = self.grid.z.to_value(u.kpc)
        
        fd = (Bpara*ne).to_value(output_units)
        
        # Converts the grids to a format compatible with the interpolator
        # (comment: this is a bit silly, but what is the native numpy alternative?)
#         points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        fd_interp = RegularGridInterpolator(points=(x[:,0,0], y[0,:,0],z[0,0,:]), values=fd)

        interp_points = np.array([coords_dict[c].to_value(u.kpc) for c in ('x','y','z')]).T

        results = fd_interp(interp_points)*output_units
        
        return results