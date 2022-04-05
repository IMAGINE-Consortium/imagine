import ImagineModels as im
import numpy as np

from imagine.fields.base_fields import MagneticField
from imagine.fields.field_factory import FieldFactory

__all__ = ['WrappedJF12', 'WrappedJF12Factory', ]


class WrappedJF12(MagneticField):
    NAME = 'WrappedJF12'
    STOCHASTIC_FIELD = False
    PARAMETER_NAMES = ['b_arm_1', 'b_arm_2', 'b_arm_3', ' b_arm_4', 'b_arm_5', 'b_arm_6',  'b_arm_7', 'b_ring', 'h_disk',
                       'w_disk', 'Bn', 'Bs', 'rn', 'rs', 'wh', 'z0', 'B0_X', 'Xtheta_const', 'rpc_X', 'r0_X', ]

    def compute_field(self, seed):
        # TODO: tolist (and hence unnecessary copy) is needed due to wrapper implementation, should change in the future

        res = self.grid.resolution
        
        x = self.grid.box[0]#.to_value()
        x = np.linspace(x[0].value, x[1].value, res[0]).tolist()
        y = self.grid.box[1]#.to_value()
        y = np.linspace(y[0].value, y[1].value, res[1]).tolist()
        z = self.grid.box[2]#.to_value()
        z = np.linspace(z[0].value, z[1].value, res[2]).tolist()
        # Creates an empty array to store the result

        wrapped_jf12 = im.JF12MagneticField()

        for key, val in self.parameters:
            if hasattr(wrapped_jf12, key):
                setattr(wrapped_jf12, key, val)

        # TODO: npasarray is needed due to wrapper implementation, should change in the future
        B = np.asarray(wrapped_jf12.evaluate_grid(grid_x=x, grid_y=y, grid_z=z))*self.UNITS

        return B


class WrappedJF12Factory(FieldFactory):
    FIELD_CLASS = WrappedJF12
    DEFAULT_PARAMETERS = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                          'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                          'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                          'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, }
    PRIORS = {}
