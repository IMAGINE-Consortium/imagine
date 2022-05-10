# %% IMPORTS
# Package imports
import numpy as np
import scipy.stats as stats
import astropy.units as u

# IMAGINE imports
from imagine.fields.base_fields import (
    MagneticField, ThermalElectronDensityField, CosmicRayElectronDensityField,
    CosmicRayElectronSpectralIndexField)

# All declaration
__all__ = ['ConstantMagneticField', 'ConstantThermalElectrons',
    'ExponentialThermalElectrons', 'RandomThermalElectrons',
    'PowerlawCosmicRayElectrons', 'ConstantCosmicRayElectrons','CRENumberDensity',
    'SpectralIndexLinearVerticalProfile']


# %% CLASS DEFINITIONS
class ConstantMagneticField(MagneticField):
    """
    Constant magnetic field

    The field parameters are:
    'Bx', 'By', 'Bz', which correspond to the fixed components
    :math:`B_x`, :math:`B_x` and :math:`B_z`.
    """

    # Class attributes
    NAME = 'constant_B'
    PARAMETER_NAMES = ['Bx', 'By', 'Bz']

    def compute_field(self, seed):
        # Creates an empty array to store the result
        B = np.empty(self.data_shape) * self.parameters['Bx'].unit
        # For a magnetic field, the output must be of shape:
        # (Nx,Ny,Nz,Nc) where Nc is the index of the component.
        # Computes Bx
        B[:, :, :, 0] = self.parameters['Bx']
        # Computes By
        B[:, :, :, 1] = self.parameters['By']
        # Computes Bz
        B[:, :, :, 2] = self.parameters['Bz']
        return B


class ConstantThermalElectrons(ThermalElectronDensityField):
    """
    Constant thermal electron density field

    The field parameters are:
    'ne', the number density of thermal electrons
    """

    # Class attributes
    NAME = 'constant_TE'
    PARAMETER_NAMES = ['ne']

    def compute_field(self, seed):
        return np.ones(self.data_shape)*self.parameters['ne']


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
    NAME            = 'exponential_disc_thermal_electrons'
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


class RandomThermalElectrons(ThermalElectronDensityField):
    """
    Thermal electron densities drawn from a Gaussian distribution

    NB This may lead to negative densities depending on the choice of
    parameters. This may be controlled with the 'min_ne' parameter
    which sets a minimum value for the density field (i.e.
    any value smaller than the minimum density is set to min_ne).

    The field parameters are: 'mean', the mean of the distribution; 'std', the
    standard deviation of the distribution; and 'min_ne', the
    aforementioned minimum density. To disable the minimum density requirement,
    it may be set to NaN.
    """

    # Class attributes
    NAME             = 'random_thermal_electrons'
    STOCHASTIC_FIELD = True
    PARAMETER_NAMES  = ['mean', 'std', 'min_ne']

    def compute_field(self, seed):
        # Converts dimensional parameters into numerical values
        # in the correct units (stats norm does not like units)
        mu = self.parameters['mean'].to_value(self.units)
        sigma = self.parameters['std'].to_value(self.units)
        minimum_density = self.parameters['min_ne'].to_value(self.units)

        # Draws values from a normal distribution with these parameters
        # using the seed provided in the argument
        distr = stats.norm(loc=mu, scale=sigma)
        result = distr.rvs(size=self.data_shape, random_state=seed)

        # Applies minimum density, if present
        if np.isfinite(minimum_density):
            result[result < minimum_density] = minimum_density

        return result << self.units  # Restores units

# ================== new from here ========================================================

class ConstantCosmicRayElectrons(CosmicRayElectronDensityField):
    """
  
    """

    # Class attributes
    NAME            = 'powerlaw_cosmicray_electrons'
    PARAMETER_NAMES = ['density',
                       'spectral_index']

    def compute_field(self, seed):
        return self.parameters['density'] * np.ones(self.grid.shape)



class PowerlawCosmicRayElectrons(CosmicRayElectronDensityField):
    """
  
    """

    # Class attributes
    NAME            = 'powerlaw_cosmicray_electrons_profile'
    PARAMETER_NAMES = ['scale_radius',
                       'scale_height',
                       'central_density',
                       'spectral_index']  
    
    def compute_field(self, seed):
        #coordinates
        z = self.grid.z
        R = self.grid.r_cylindrical
        #calculate density
        Re = self.parameters['scale_radius']
        he = self.parameters['scale_height']
        nc = self.parameters['central_density']
        nCRE = nc*np.exp(-R/Re)*np.exp(-np.abs(z/he))
        return nCRE

class CRENumberDensity(CosmicRayElectronDensityField):
    """ 
    
    """    
    
    # Class attributes
    NAME            = 'cosmicray_electron_numberdensity_profile'
    PARAMETER_NAMES = ['scale_radius',
                       'scale_height',
                       'central_density']  
    
    def compute_field(self, seed):
        #coordinates
        z = self.grid.z
        R = self.grid.r_cylindrical
        #calculate density
        Re = self.parameters['scale_radius']
        he = self.parameters['scale_height']
        nc = self.parameters['central_density']
        nCRE = nc*np.exp(-R/Re)*np.exp(-np.abs(z/he))
        return nCRE

class SpectralIndexLinearVerticalProfile(CosmicRayElectronSpectralIndexField):
    
    """
    CRE spectral hardening as a function of distance to the Galatic disk
    """    
    
    # Class attributes
    NAME            = 'spectral_index_linear_vertical_profile'
    PARAMETER_NAMES = ['soft_index', 'hard_index', 'slope']  
    
    def compute_field(self, seed):
        # coordinates and parameters
        z = self.grid.z
        s = self.parameters['soft_index'] # number like -4
        h = self.parameters['hard_index'] # number like -2.2
        slope = self.parameters['slope']
        # calculate spectral index
        alpha = s + slope*np.abs(z)
        alpha[alpha>h] = h
        return np.full(shape=self.grid.shape,fill_value=alpha)*u.dimensionless_unscaled

# ========================= Not functional from here ========================

class CosmicRayElectronEnergyDensity(CosmicRayElectronDensityField):
    """
    4D grid 
    
    """
    
    # Class attributes
    NAME            = 'arbitrary_spectrum_cosmicray_electrons'
    PARAMETER_NAMES = ['scale_radius',
                       'scale_height',
                       'central_density']  
    
    
    def __init__(self, grid, parameters=None):
        super().__init__(grid)
        if parameters is not None:
            self.parameters = parameters
    
    
        if callable(spectral_index):           
            self.spectral_index_grid = np.zeros(self.grid.shape)
            
            # coordinates
            x = grid.x[:,0,0].to_value(u.kpc)
            y = grid.y[0,:,0].to_value(u.kpc)
            z = grid.z[0,0,:].to_value(u.kpc)          
            print(np.shape(x))            
            
            # caculate full grid of spectral indices
            for i in range(len(x)):
                for j in range(len(y)):
                    for k in range(len(z)):
                        #print(x[i],y[j],z[k])
                        #print(spectral_index(x[i],y[j],z[k]),'\n')
                        self.spectral_index_grid[i,j,k] = spectral_index(x[i],y[j],z[k])
            
            # after calculating this grid as a new class attribute set constant index to None
            self.parameters['spectral_index'] = None     
    
    def compute_field(self, seed):
        R = self.grid.r_cylindrical
        z = self.grid.z
        Re = self.parameters['scale_radius']
        he = self.parameters['scale_height']
        n0 = self.parameters['central_density']
        return n0*np.exp(-R/Re)*np.exp(-np.abs(z/he)) 





class BrokenPowerlawCosmicRayElectrons(CosmicRayElectronDensityField):
    def compute_field(self, seed):
        return 





















