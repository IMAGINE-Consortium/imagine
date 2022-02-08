#%% IMPORTS
# Built-in imports
import logging as log
from os import path
import tempfile

# Package imports
import astropy.units as u
import hampyx
from hampyx import Hampyx
import numpy as np

# IMAGINE imports
import imagine as img
from imagine.simulators import Simulator
from imagine.observables import Masks

# All declaration
__all__ = ['synchrotron']


#%% CLASS DEFINITIONS

from numpy.polynomial import Polynomial

class SynchrotronEmissivitySimulator(Simulator):
    """
    Simulator for synchrotron emissivity 
    
    
    """
    
    # Class attributes
    SIMULATED_QUANTITIES = ['sync']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'cosmic_ray_electron_density']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self, measurements, observing_frequency, observer_position):
        # Send the Measurements to the parent class
        super().__init__(measurements) 
        # Stores class-specific attributes
        self.observing_frequency = observing_frequency        
        
        # After setting observer position make unit vectorgrid        
        self.xobs = observer_position[0]        
        self.yobs = observer_position[1]
        self.zobs = observer_position[2]
        #vectors   = self.grid - position
        #self.unitvectors = vectors/norm_vectors
        
    def _polyfitF(x):
        """
        The integration of the modified besselfunction of the second kind was
        was fitted with a 30 degree polynomial. Other parametric fits could
        be implemented here.
        The numerical stability of the integration and the quality of the 
        fit was investigated in [add ref].
        """
        coef = [-6.70416283e-01,  1.61018924e+00, -5.63219709e-01,  4.04961264e+00,
                3.65149319e+01, -1.98431222e+02, -1.19193545e+03,  4.17965143e+03,
                1.99241293e+04, -4.98977562e+04, -1.99974183e+05,  3.72435079e+05,
                1.30455499e+06, -1.85036218e+06, -5.81373137e+06,  6.35262141e+06,
                1.82406280e+07, -1.53885501e+07, -4.09471882e+07,  2.64933708e+07,
                6.60352613e+07, -3.22140638e+07, -7.58569209e+07,  2.70300472e+07,
                6.05473150e+07, -1.48865555e+07, -3.19002452e+07,  4.84145538e+06,
                9.97173698e+06, -7.04507542e+05, -1.40022630e+06]
        domain = [-8.,  2.]
        polyfit_integral = Polynomial(coef, domain)        
        return 10**polyfit_integral(np.log10(x)) 
        
    def _critical_freq(gamma):
        return (3*electron*Bper/(2*me*c) * (gamma)**2).decompose(bases=u.cgs.bases)

    def _critical_gamma(wc):
        return np.sqrt(wc * (2*me*c)/(3*electron*Bper)).decompose()
    
    def _emission_power_density(gamma, wobs, Bper):
        F  = polyfitF(wobs/critical_freq(gamma))
        return ((np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * F).decompose(bases=u.cgs.bases)
    
    def _integrate_total_emissivity(self):
        # Calculation of emissivity requires integration over gamma
        energygrid = 10**np.linspace(7,13,int(1e4))*eV
        gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)       
        
        wobs = self.observing_frequency
        Bper = self.fields['magnetic_field'] #project on unitvectors
        ncre = self.fields['cosmic_ray_electron_density']
        dPw  = emission_power_density(gammas, wobs, Bper).decompose(bases=u.cgs.bases)
        dg   = np.ediff1d(gammas)
        
        # implement midpoint
        dJleft  = 0.5*np.sum((ncre[:-1]*dPw[:-1]*dg).decompose(bases=u.cgs.bases))
        dJright = 0.5*np.sum((ncre[1:]*dPw[1:]*dg).decompose(bases=u.cgs.bases))
        return (dJleft+dJright)/2
    
    def simulate(self):
        """
        Simulating the grid of emissivities is handeld differently depending on
        the type of cosmic ray density models specified.        
        """
        
        Bper = self.fields['magnetic_field'] #project on unitvectors
        ncre = self.fields['cosmic_ray_electron_density']
        
        if self.fields['cosmic_ray_electron_density'].NAME == 'powerlaw_cosmicray_electrons':
            # loop through all gridpoints
            emissivity_grid[,,] = spectral_emissivity(self.fields['cosmic_ray_electron_density'].alpha)
        
        elif self.fields['cosmic_ray_electron_density'].NAME == 'broken_powerlaw_cosmicray_electrons':
            emissivity_grid[] = _integrate_total_emissivity()
        
                
        
        return emissivity_grid