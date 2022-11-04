import numpy as np
import astropy.units as u

from imagine.fields.TOGOModel import Model
from imagine.fields.TOGOsimple.TOGOSimpleMagnetic import ConstantMagneticField
from imagine.fields.TOGOsimple.TOGOSimpleThermalElectron import ConstantThermalElectron

from imagine.fields.grid import UniformGrid

parameters = {'Bx': 1.*u.microgauss, 'By': -1.*u.microgauss, 'Bz': 0.*u.microgauss, 'ne': 5*u.cm**(-3)}

xlims = [0,4]*u.kpc; ylims = [1,2]*u.kpc; zlims = [1,1]*u.kpc
grid = UniformGrid([xlims, ylims, zlims], [5,2,1])


b1 = ConstantMagneticField(grid)
b2 = ConstantMagneticField(grid)
n1 = ConstantThermalElectron(grid)

b3 = b1 + b2
b4 = b1.vec_abs()

bn1 = b4*n1
print(b1.compute_model(parameters))
print(n1.compute_model(parameters))
print(b3.compute_model(parameters))
print(bn1.compute_model(parameters))
