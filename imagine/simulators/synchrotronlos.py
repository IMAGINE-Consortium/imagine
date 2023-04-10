"""
This is built on the master thesis of Youri Sloots.
"""


# Simulator dependancies
import numpy as np
from imagine.simulators import Simulator
from scipy.special import gamma as gammafunc
from .LOSresponse import build_nifty_los, apply_response


import astropy.units as u

from astropy import constants as cons
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
me    = cons.m_e.cgs
c     = cons.c.cgs
kb    = cons.k_B.cgs
electron = cons.e.gauss

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B


__all__ = ['SpectralSynchrotronEmissivitySimulator']

#%% Define the Simulator class

class SpectralSynchrotronEmissivitySimulator(Simulator):
    """
    Simulator for Galactic synchrotron emissivity.
    
    Requires the user to define observing geometry in the config dictionary    
    
    To run the default __call__ method one requires to input:
    - A 3D magnetic_field instance
    - A cosmic_ray_electron_density instance: 3D scalar density grid and constant spectral_index attribute
    
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
    domain : 
        Nifty version of grid, used in the integration
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
    SIMULATED_QUANTITIES = ['average_los_brightness']
    REQUIRED_FIELD_TYPES = ['magnetic_field','cosmic_ray_electron_density']
    OPTIONAL_FIELD_TYPES = ['cosmic_ray_electron_spectral_index']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self,measurements, sim_config={'grid':None,'observer':None,'dist':None,'e_dist':None,'lat':None,'lon':None,'FB':None}):
        
        # print("Initializing SynchtrotronEmissivitySimulator")

        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # Write assert and test functions to make sure stuf is correct
        # - grid.unit == observer.unit
        # - grid = instance of cartesian grid
        # - all HII regions should be located within the simulation box        
        
        
        # Asses field types and set data-acces function
        #self.get_field_data()        
        
        # Stores class-specific attributes (and for now double definitions)
        for key in measurements.keys(): self.observing_frequency = key[1] * GHz
        grid = sim_config['grid'] # unpack for readability
        

        observer = sim_config['observer']
        # Get lines of sight from data
        lat        = sim_config['lat'].to(u.rad)
        lon        = sim_config['lon'].to(u.rad)
        dist    = sim_config['dist'].to(u.kpc)
        e_dist  = sim_config['e_dist'].to(u.kpc)
        behind     = np.where(np.array(sim_config['FB'])=='B')
        
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
    
        
 
    

    
    def _spectral_integralF(self, mu):
        return 2**(mu+1) / (mu+2) * gammafunc(mu/2 + 7./3) * gammafunc(mu/2 + 2./3)    
    
    def _spectral_total_emissivity(self, Bper, ncre):
        # Check constant units and remove them for the calculation
        e     = (cons.e).gauss / (cons.e).gauss.unit
        me    = cons.m_e.cgs / cons.m_e.cgs.unit
        c     = cons.c.cgs / cons.c.cgs.unit
        # Check argument units and remove them for the calculation        
        vobs  = self.observing_frequency.to(1/u.s)*u.s
        ncre  = ncre.to(u.cm**-3)*u.cm**3
        Bper  = Bper.to(u.G) / u.G
        # Handle two spectral index cases:
	# -> fieldlist is not provided on initialization so we opt for a runtime check of alpha type
        try: # alpha is a constant spectral index globally 
            alpha = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        except: pass
        try: # alpha is an instance of a 3D scalar field
            alpha = self.fields['cosmic_ray_electron_spectral_index']
        except: pass
        # Calculate emissivity grid
        fraction1 = (np.sqrt(3)*e**3*ncre/(8*np.pi*me*c**2))
        fraction2 = (4*np.pi*vobs*me*c/(3*e))
        integral  = self._spectral_integralF( (-alpha-3)/2 )
        emissivity = fraction1 * fraction2**((1+alpha)/2) * Bper**((1-alpha)/2) * integral
        assert emissivity.unit == u.dimensionless_unscaled
        # Return emissivty and restore correct units
        return fraction1*fraction2**((1+alpha)/2)*Bper**((1-alpha)/2)*integral * u.kg/(u.m*u.s**2)
    
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
        ncre_grid = self.fields['cosmic_ray_electron_density']  # in units cm^-3
        B_grid    = self.fields['magnetic_field'] # fixing is now done inside emissivity calculation
        # Project to perpendicular component to line of sight
        Bperp_amplitude_grid = self._project_to_LOS(B_grid, parallel=False, return_only_amplitude=True)        
        # Calculate grid of emissivity values
        emissivity_grid = self._spectral_total_emissivity(Bperp_amplitude_grid, ncre_grid)
        # Do the los integration on the domain defined in init with the new emissivity grid
        HII_LOSemissivities = apply_response(self.response, emissivity_grid)
        HII_LOSemissivities *= emissivity_grid.unit * u.kpc # restore units: domain is assumed to be in kpc
        # Need units to be in K/kpc, average brightness temperature allong the line of sight
        HII_LOSbrightness = c**2/(2*kb*self.observing_frequency**2)*HII_LOSemissivities/self.distances
        return HII_LOSbrightness

