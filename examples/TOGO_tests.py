import numpy as np
import astropy.units as u

from imagine.fields.TOGOfield import Model
from imagine.fields.TOGOsimple.TOGOSimpleMagnetic import ConstantMagneticField

from imagine import UniformGrid

parameters = {'Bx': 1., 'By': -1., 'Bz': 0.}

xlims = [0,4]*u.kpc; ylims = [1,2]*u.kpc; zlims = [1,1]*u.kpc
grid = UniformGrid([xlims, ylims, zlims], [5,2,1])


c1 = ConstantMagneticField(grid)

print(c1.compute_field(parameters))
