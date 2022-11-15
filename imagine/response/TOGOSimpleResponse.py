import numpy as np
import astropy.units as apu

from .TOGOResponse import Response
from ..fields.TOGOModel import Model
from ..grid.TOGOGrid import ParameterSpace, UniformGrid


class SimpleIntegrator(Response):

    def __init__(self, input_grid, direction, name=None):
        if not isinstance(input_grid, UniformGrid):
            raise TypeError()
        nlos = 1
        for j, res in enumerate(input_grid.resolution):
            if j != direction:
                nlos *= res

        output_space = ParameterSpace(nlos, 'IntegratorOutput')
        super().__init__(input_grid, None, output_space, call_by_method=True)

        self.direction = direction
        box_limits = input_grid.box[direction]
        self.dx = (box_limits[1] - box_limits[0])/input_grid.resolution[direction]

    def compute_model(self, field):
        if not isinstance(field, np.ndarray):
            raise TypeError('Imagine.SimpleIntegrator: Can only be applied on numpy arrays, was applied on {}'.format(type(field)))
        return np.cumsum(field, axis=self.direction).flatten()*self.dx
