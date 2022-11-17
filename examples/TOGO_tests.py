import numpy as np
import astropy.units as u

from imagine.fields.TOGOModel import Model
from imagine.fields.TOGOsimple.TOGOSimpleMagnetic import ConstantMagneticField
from imagine.fields.TOGOsimple.TOGOSimpleThermalElectron import ConstantThermalElectron, ExponentialThermalElectron
from imagine.response.TOGOSimpleResponse import SimpleIntegrator
from imagine.priors.TOGOPrior import Prior, MultivariatePrior


from imagine.grid.TOGOGrid import UniformGrid, ParameterSpace

xlims = [0, 4] * u.kpc
ylims = [1, 2] * u.kpc
zlims = [1, 1] * u.kpc

grid = UniformGrid([xlims, ylims, zlims], [5, 2, 1])

bx_prior = Prior('uniform', 'Bx', u.microgauss)
by_prior = Prior('norm', 'By', u.microgauss, loc=1)
bz_prior = Prior('t', 'Bz', u.microgauss, loc=-3, scale=2, df=1)
ne_prior = Prior('lognorm', 'ne', u.cm**(-3), loc=1, s=1)
cd_prior = Prior('lognorm', 'central_density', u.cm**(-3), loc=-1, s=3)
sr_prior = Prior('expon', 'scale_radius', u.kpc, loc=-1)
sh_prior = Prior('expon', 'scale_height', u.kpc, loc=2)


full_prior = MultivariatePrior({'Bx': bx_prior, 'By': by_prior, 'Bz': bz_prior, 'ne': ne_prior,
                                'central_density': cd_prior, 'scale_radius': sr_prior, 'scale_height': sh_prior
                                })


parameters = full_prior.draw_sample()
print(parameters)

b1 = ConstantMagneticField(grid)
b2 = b1.radial_component()
print(b2(parameters))
print('b1 ', type(b1))
print('b2 ', type(b2))

n1 = ConstantThermalElectron(grid)
n2 = ExponentialThermalElectron(grid)

print('n1 ', type(n1))
print('n2 ', type(n2))

n3 = n1 + n2
print('n3 ', type(n3))

bn1 = b2*n3

print('bn1 ', type(bn1))

r1 = SimpleIntegrator(grid, 0)

s1 = r1@bn1

# print(bn1(parameters))
print(r1(bn1(parameters)))
print(s1(parameters))

fullmodel = s1@full_prior

print('fullmodel ', type(fullmodel))


#print(b1.compute_model(parameters))
#print(n1.compute_model(parameters))
#print(b3.compute_model(parameters))
#print(bn1.compute_model(parameters))
