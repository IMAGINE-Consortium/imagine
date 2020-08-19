from imagine.fields import ThermalElectronDensityField
import numpy as np
import MY_GALAXY_MODEL # Substitute this by your own code


class ThermalElectronsDensityTemplate(ThermalElectronDensityField):
    """ Here comes the description of the electron density model """

    # Class attributes
    NAME = 'name_of_the_thermal_electrons_field'

    # Is this field stochastic or not. Only necessary if True
    STOCHASTIC_FIELD = True
    # If there are any dependencies, they should be included in this list
    DEPENDENCIES_LIST = []
    # List of all parameters for the field
    PARAMETER_NAMES = ['Parameter_A', 'Parameter_B']

    def compute_field(self, seed):
        # If this is an stochastic field, the integer `seed `must be
        # used to set the random seed for a single realisation.
        # Otherwise, `seed` should be ignored.

        # The coordinates can be accessed from an internal grid object
        x_coord = self.grid.x
        y_coord = self.grid.y
        z_coord = self.grid.y
        # Alternatively, one can use cylindrical or spherical coordinates
        r_cyl_coord = self.grid.r_cylindrical
        r_sph_coord = self.grid.r_spherical
        theta_coord = self.grid.theta
        phi_coord = self.grid.phi

        # One can access the parameters supplied in the following way
        param_A = self.parameters['Parameter_A']
        param_B = self.parameters['Parameter_B']

        # Now you can interface with previous code or implement here
        # your own model for the thermal electrons distribution.
        # Returns the electron number density at each grid point
        # in units of (or convertible to) cm**-3
        return MY_GALAXY_MODEL.compute_ne(param_A, param_B,
                                          r_sph_coord, theta_coord, phi_coord,
                                          # If the field is stochastic
                                          # it can use the seed
                                          # to generate a realisation
                                          seed)
