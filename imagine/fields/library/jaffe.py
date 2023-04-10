import ImagineModels as models
import numpy as np
import astropy.units as u
import astropy as ap

from imagine.fields.base_fields import MagneticField
from imagine.fields.field_factory import FieldFactory

__all__ = ['Jaffe', 'JaffeFactory', ]


class Jaffe(MagneticField):
    NAME = 'Jaffe'
    STOCHASTIC_FIELD = False
    PARAMETER_NAMES = ["disk_amp", "disk_z0", "halo_amp", "halo_z0", "r_inner", "r_scale", "r_peak", 
                      "ring_amp", "ring_r", "bar_amp", "bar_a", "bar_b", "bar_phi0", 
                       "arm_r0", "arm_z0", "arm_phi1", "arm_phi2", "arm_phi3", "arm_phi4", "arm_amp1", 
                      "arm_amp2", "arm_amp3", "arm_amp4", "arm_pitch", 
                      "comp_c", "comp_d", "comp_r", "comp_p",]

    DEFAULT_UNITS =  {"disk_amp": u.microgauss, "disk_z0":  u.kpc, "halo_amp": u.microgauss,  
                      "halo_z0":  u.kpc, "r_inner":  u.kpc, "r_scale":  u.kpc, "r_peak":  u.kpc, 
                      
                      "ring_amp": u.microgauss, "ring_r":  u.kpc, "bar_amp": u.microgauss, 
                      "bar_a":  u.kpc, "bar_b":  u.kpc, "bar_phi0": u.deg, 
                      
                      "arm_r0":  u.kpc, "arm_z0":  u.kpc, "arm_phi1": u.deg, "arm_phi2": u.deg, 
                      "arm_phi3": u.deg, "arm_phi4": u.deg, "arm_amp1": u.microgauss, 
                      "arm_amp2": u.microgauss, "arm_amp3": u.microgauss, "arm_amp4": u.microgauss, 
                      "arm_pitch": u.deg, 
                      
                      "comp_c": None, "comp_d":  u.kpc, "comp_r":  u.kpc, "comp_p": None,
                    }

    def compute_field(self, seed):
        
        res = self.grid.resolution

        x = self.grid.box[0]#.to_value()
        x = np.linspace(x[0].value, x[1].value, res[0])
        y = self.grid.box[1]#.to_value()
        y = np.linspace(y[0].value, y[1].value, res[1])
        z = self.grid.box[2]#.to_value()
        z = np.linspace(z[0].value, z[1].value, res[2])
        # Creates an empty array to store the result

        jaffe = models.JaffeMagneticField()

        for key, val in self.parameters.items():
            if hasattr(jaffe, key):
                if isinstance(val, ap.Quantity):
                    val = val.to(self.DEFAULT_UNITS[key]).value
                setattr(jaffe, key, val)

        B = jaffe.on_grid(grid_x=x, grid_y=y, grid_z=z)*self.UNITS
        #print("Found {} nans inside JF12 field".format(np.sum(np.isnan(B))))

        # Test B for nans
        #if np.sum(np.isnan(B)) != 0:
            #print("Setting {} nans to 0".format(np.sum(np.isnan(B))))
            #B[np.isnan(B)] = 0*self.UNITS

        return B


class JaffeFactory(FieldFactory):
    FIELD_CLASS = Jaffe
    DEFAULT_PARAMETERS = {}
    PRIORS = {}
