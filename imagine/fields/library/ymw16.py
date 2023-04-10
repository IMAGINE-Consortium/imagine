import ImagineModels as im
import numpy as np
import astropy.units as u
import astropy as ap

from imagine.fields.base_fields import MagneticField
from imagine.fields.field_factory import FieldFactory

__all__ = ['YMW16', 'YMW16Factory', ]


class YMW16(MagneticField):
    NAME = 'YMW16'
    STOCHASTIC_FIELD = False
    PARAMETER_NAMES = ['b_arm_1', 'b_arm_2', 'b_arm_3', 'b_arm_4', 'b_arm_5', 'b_arm_6',  'b_arm_7', 'b_ring', 'h_disk',
                       'w_disk', 'Bn', 'Bs', 'rn', 'rs', 'wh', 'z0', 'B0_X', 'Xtheta_const', 'rpc_X', 'r0_X', ]

    DEFAULT_UNITS =  {'b_arm_1': u.microgauss, 'b_arm_2': u.microgauss, 'b_arm_3': u.microgauss, 'b_arm_4': u.microgauss, 'b_arm_5': u.microgauss,
                      'b_arm_6': u.microgauss, 'b_arm_7': u.microgauss, 'b_ring': u.microgauss, 'h_disk': u.kpc, 'w_disk': u.kpc,
                      'Bn': u.microgauss, 'Bs': u.microgauss, 'rn':  u.kpc, 'rs': u.kpc, 'wh':  u.kpc, 'z0':  u.kpc, 'B0_X': u.microgauss,
                      'Xtheta_const': u.rad, 'rpc_X': u.kpc, 'r0_X': u.kpc, }

    def compute_field(self, seed):
        
        res = self.grid.resolution

        x = self.grid.box[0]#.to_value()
        x = np.linspace(x[0].value, x[1].value, res[0])
        y = self.grid.box[1]#.to_value()
        y = np.linspace(y[0].value, y[1].value, res[1])
        z = self.grid.box[2]#.to_value()
        z = np.linspace(z[0].value, z[1].value, res[2])
        # Creates an empty array to store the result

        model = im.YMW16()

        for key, val in self.parameters.items():
            if hasattr(model, key):
                if isinstance(val, ap.Quantity):
                    val = val.to(self.DEFAULT_UNITS[key]).value
                setattr(model, key, val)

        m = model.evaluate_grid(grid_x=x, grid_y=y, grid_z=z)*self.UNITS

        return m


class YMW16Factory(FieldFactory):
    FIELD_CLASS = YMW16
    DEFAULT_PARAMETERS = { }
    PRIORS = {}
