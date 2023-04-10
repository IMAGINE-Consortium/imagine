"""
This is built on the synchtrotron los simulator of Youri Sloots.
"""


# Simulator dependancies
import numpy as np
from imagine.simulators import Simulator
from .LOSresponse import build_nifty_los, apply_response


import astropy.units as u

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B


__all__ = ['RMSimulator']

#%% Define the Simulator class

class RMSimulator(Simulator):
    """
    Simulator for Galactic RMs.
    
    Requires the user to define observing geometry in the config dictionary    
    
    To run the default __call__ method one requires to input:
    - A 3D magnetic_field instance
    - A thermal_electron_density instance: 3D scalar density grid
    
    Parameters
    ----------
    measurements : imagine.Measurements
        An observables dictionary containing the set of measurements that will be
        used to prepare the mock observables
    sim_config : 
        A dictionary detailing the observing geometry

    Attributes
    ----------
    grid : imagine.Basegrid
        Grid object where the fields were evaluated
    unitvector_grid : numpy.ndarray
        3D vector field containing the unitvectors pointing towards observer, used 
        for projecting to perpendicular magnetic field component
    fields : dict
        Dictionary containing field types as keys and the sum of evaluated fields
        as values
    observables : list
        List of Observable keys
    output_units : astropy.units.Unit
        Output units used in the simulator
    """

    
    # Class attributes
    SIMULATED_QUANTITIES = ['fd']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'thermal_electron_density']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self, measurements, sim_config={'grid':None, 'observer':None, 'dist':None, 'e_dist':None, 'lat':None,' lon':None}):
        
        # print("Initializing RMSimulator")

        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # unpack for readability
        
        grid = sim_config['grid'] 

        observer = sim_config['observer']

        lat        = sim_config['lat'].to(u.rad)
        lon        = sim_config['lon'].to(u.rad)
        dist    = sim_config['dist'].to(u.kpc)
        e_dist  = sim_config['e_dist'].to(u.kpc)
        behind     = np.where(np.full(len(dist), False))
        
        self.response = build_nifty_los(grid, behind, u.kpc, observer, dist, lon, lat, e_dist)
        
        self.distances = dist # need for average emissivities
        
        unitvectors = []
        for x,y,z in zip(grid.x.ravel()/u.kpc, grid.y.ravel()/u.kpc, grid.z.ravel()/u.kpc):
            v = np.array([x,y,z]) - observer/u.kpc
            normv = np.linalg.norm(v)
            if normv == 0: # special case where the observer is inside one of the grid points
                unitvectors.append(v)
            else:
                unitvectors.append(v/normv)
        # Save unitvector grid as class attribute
        self.unitvector_grid = np.reshape(unitvectors, tuple(grid.resolution)+(3,))
    
        
    def _project_to_LOS(self, vectorfield, parallel=True, return_only_amplitude=False):
        """
        This function takes in a 3D vector field and uses the initialized unitvector_grid
        to project each vector on the unit vector perpendicular to the los to that position.
        """
        v_return      = np.zeros(np.shape(vectorfield)) * vectorfield.unit
        amplitudes      = np.sum(vectorfield * self.unitvector_grid, axis=3)
        v_return[:,:,:,0]  = amplitudes * self.unitvector_grid[:,:,:,0]
        v_return[:,:,:,1]  = amplitudes * self.unitvector_grid[:,:,:,1]
        v_return[:,:,:,2]  = amplitudes * self.unitvector_grid[:,:,:,2]
        if not parallel:
            v_return = vectorfield - v_return
        if return_only_amplitude:
            v_return = np.sqrt(np.sum(v_return*v_return, axis=3))
        return v_return
    
    
    
    def simulate(self, key, coords_dict, realization_id, output_units): 
        # Acces field data
        nth = self.fields['thermal_electron_density']  # in units cm^-3
        Bf    = self.fields['magnetic_field'] # fixing is now done inside emissivity calculation
        # Project to perpendicular component to line of sight
        Bpar = self._project_to_LOS(Bf, parallel=True, return_only_amplitude=False)        
        # Calculate grid of emissivity values
        nth  = nth.to(u.cm**-3)*u.cm**3
        Bpar  = Bpar.to(u.G) / u.G
        rm_source = 0.812*(Bpar*nth)*ugauss_B
        # Do the los integration 
        rms = apply_response(self.response, rm_source)
        rms *= u.rad/(u.m**2) # restore units
        return rms

