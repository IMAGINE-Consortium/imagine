from imagine import MagneticField
import numpy as np
import MY_GMF_MODEL # Substitute this by your own code

@icy
class MagneticFieldTemplate(MagneticField):
    """ Here comes the description of the magnetic field model """
    @property
    def field_checklist(self):
        # This property returns a dictionary with all the 
        # available parameters as keys
        return {'Parameter_A': None, 'Parameter_B': None, ...}
    
    def get_field(parameters):
        # The coordinates can be accessed from an internal grid object
        x_coord = self.grid.x
        y_coord = self.grid.y
        z_coord = self.grid.y
        # Alternatively, one can use cylindrical or spherical coordinates
        r_cyl_coord = self.grid.r_cylindrical
        r_sph_coord = self.grid.r_spherical
        theta_coord = self.grid.theta; phi_coord = self.grid.phi
        
        # One can access the parameters supplied in the following way
        param_A = self.parameters['Parameter_A']
        param_B = self.parameters['Parameter_B']
        
        # Now one can interface with previous code, or implement a 
        # particular magnetic field
        Bx, By, Bz = MY_GMF_MODEL.compute_B(param_A, param_B, 
                                            x_coord, y_coord, z_coord)
        
        # Creates an empty output magnetic field array with
        # the correct shape
        MF_array = np.empty(self.data_shape)
        # and saves the pre-computed components
        MF_array[:,:,:,0] = Bx
        MF_array[:,:,:,1] = By
        MF_array[:,:,:,2] = Bz
        
        return MF_array
