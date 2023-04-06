import ImagineModels as im
import numpy as np
import astropy.units as u

from imagine.fields.TOGOBaseModels import VectorField

__all__ = ['JF12Regular', ]

gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]


class JF12Regular(VectorField):
    PARAMETER_NAMES = ['b_arm_1', 'b_arm_2', 'b_arm_3', ' b_arm_4', 'b_arm_5', 'b_arm_6',  'b_arm_7', 'b_ring', 'h_disk',
                       'w_disk', 'Bn', 'Bs', 'rn', 'rs', 'wh', 'z0', 'B0_X', 'Xtheta_const', 'rpc_X', 'r0_X', ]

    def __init__(self, grid, use_default_params_where_necessary=False):
        parameter_def = {'b_arm_1': gauss_B, 'b_arm_2': gauss_B, 'b_arm_3': gauss_B, 'b_arm_4': gauss_B,
                         'b_arm_5': gauss_B, 'b_arm_6': gauss_B, 'b_arm_7': gauss_B, 'b_ring': gauss_B,
                         'h_disk': u.kpc, 'w_disk': u.kpc, 'Bn': gauss_B, 'Bs': gauss_B, 'rn': u.kpc, 'rs': u.kpc,
                         'wh': u.kpc, 'z0': u.kpc, 'B0_X': gauss_B, 'Xtheta_const': u.deg, 'rpc_X': u.kpc,
                         'r0_X': u.kpc, }

        super().__init__(grid, parameter_def, gauss_B, 3, call_by_method=True)

    def compute_model(self, parameters):
        # TODO: tolist (and hence unnecessary copy) is needed due to wrapper implementation, should change in the future

        res = self.grid.resolution

        x = self.grid.box[0]  # .to_value()
        x = np.linspace(x[0].value, x[1].value, res[0]).tolist()
        y = self.grid.box[1]  # .to_value()
        y = np.linspace(y[0].value, y[1].value, res[1]).tolist()
        z = self.grid.box[2]  # .to_value()
        z = np.linspace(z[0].value, z[1].value, res[2]).tolist()
        # Creates an empty array to store the result

        wrapped_jf12 = im.JF12MagneticField()

        for key, val in parameters.items():
            if hasattr(wrapped_jf12, key):
                setattr(wrapped_jf12, key, val)
            else:
                raise KeyError()

        # TODO: npasarray is needed due to wrapper implementation, should change in the future
        return np.asarray(wrapped_jf12.evaluate_grid(grid_x=x, grid_y=y, grid_z=z)) << self.unit
        #print("Found {} nans inside JF12 field".format(np.sum(np.isnan(B))))

        # Test B for nans
        #if np.sum(np.isnan(B)) != 0:
        #print("Setting {} nans to 0".format(np.sum(np.isnan(B))))
        #B[np.isnan(B)] = 0*self.UNITS
