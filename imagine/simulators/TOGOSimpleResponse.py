import numpy as np
import astropy.units as apu

from .TOGOResponse import Response
from ..fields.TOGOModel import Model


class Integrator(Response):

    REQUIRED_FIELD_TYPES = ['Uniform']

    def __init__(self, input_grid, direction):
        output_grid =
        super.__init__(input_grid)


    def simulate(field):
        if not is isinstance(field, Model):
            raise TypeError()
        
