import numpy as np
import astropy.units as apu

from .TOGOResponse import Response
from ..fields.TOGOModel import Model
from ..fields.TOGOGrid import ParameterSpace, UniformGrid


class SimpleIntegrator(Response):

    def __init__(self, input_grid, direction, name=None):
        if not isinstance(input_grid, UniformGrid):
            raise TypeError()
        nlos =  1
        for j, res in enumerate(input_grid.resolution):
            if j != direction:
                nlos *= res

        output_space = ParameterSpace(nlos, 'IntegratorOutput')
        super.__init__(input_grid, output_space)

        self.direction = direction
        box_limits = input_grid.box[direction]
        self.dx = (box_limits[1] - box_limits[1])/input_grid.resolution[direction]


    def simulate(self, field):
        if not isinstance(field, np.array):
            raise TypeError()
        return np.cumsum(field, axis=self.direction).flatten()*self.dx
