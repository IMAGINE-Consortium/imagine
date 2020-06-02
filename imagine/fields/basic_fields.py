from imagine import MagneticField, ThermalElectronDensityField
from imagine.tools.icy_decorator import icy
import numpy as np
import scipy.stats as stats

@icy
class ConstantMagneticField(MagneticField):
    """
    Constant magnetic field

    The field parameters are:
    'Bx', 'By', 'Bz', which correspond to the fixed components
    :math:`B_x`, :math:`B_x` and :math:`B_z`.
    """
    field_name = 'constantB'
    stochastic_field = False

    @property
    def field_checklist(self):
        return {'Bx': None, 'By': None, 'Bz': None}

    def compute_field(self, seed):
        # Creates an empty array to store the result
        B = np.empty(self.data_shape) * self.parameters['Bx'].unit
        # For a magnetic field, the output must be of shape:
        # (Nx,Ny,Nz,Nc) where Nc is the index of the component.
        # Computes Bx
        B[:,:,:,0] = self.parameters['Bx']*np.ones(self.grid.shape)
        # Computes By
        B[:,:,:,1] = self.parameters['By']*np.ones(self.grid.shape)
        # Computes Bz
        B[:,:,:,2] = self.parameters['Bz']*np.ones(self.grid.shape)
        return B


@icy
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

    field_name = 'exponential_disc_thermal_electrons'
    stochastic_field = False

    @property
    def field_checklist(self):
        return {'central_density' : None,
                'scale_radius' : None,
                'scale_height' : None}

    def compute_field(self, seed):
        R = self.grid.r_cylindrical
        z = self.grid.z
        Re = self.parameters['scale_radius']
        he = self.parameters['scale_height']
        n0 = self.parameters['central_density']

        return n0*np.exp(-R/Re)*np.exp(-np.abs(z/he))

@icy
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

    field_name = 'random_thermal_electrons'
    stochastic_field = True

    @property
    def field_checklist(self):
        return {'mean' : None, 'std' : None, 'min_ne': None}

    def compute_field(self, seed):
        # Converts dimensional parameters into numerical values
        # in the correct units (stats norm does not like units)
        mu = self.parameters['mean'].to_value(self.field_units)
        sigma = self.parameters['std'].to_value(self.field_units)
        minimum_density = self.parameters['min_ne'].to_value(self.field_units)

        # Draws values from a normal distribution with these parameters
        # using the seed provided in the argument
        distr = stats.norm(loc=mu, scale=sigma)
        result = distr.rvs(size=self.data_shape, random_state=seed)

        # Applies minimum density, if present
        if np.isfinite(minimum_density):
            result[result<minimum_density] = minimum_density

        return result << self.field_units # Restores units
