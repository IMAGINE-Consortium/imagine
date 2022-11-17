import numpy as np
import astropy.units as u

from imagine.fields.TOGOModel import Model
from imagine.fields.TOGOsimple.TOGOSimpleMagnetic import ConstantMagneticField
from imagine.fields.TOGOsimple.TOGOSimpleThermalElectron import ConstantThermalElectron, ExponentialThermalElectron
from imagine.response.TOGOSimpleResponse import SimpleIntegrator
from imagine.priors.TOGOPrior import UnivariatePrior, MultivariatePrior
from imagine.likelihoods.likelihood import TOGOGaussianLogLikelihood


from imagine.grid.TOGOGrid import UniformGrid, ParameterSpace


gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]
xlims = [0, 4] * u.kpc
ylims = [1, 2] * u.kpc
zlims = [1, 1] * u.kpc

grid = UniformGrid([xlims, ylims, zlims], [5, 2, 1])

bx_prior = UnivariatePrior('uniform', 'Bx', gauss_B)
by_prior = UnivariatePrior('norm', 'By', gauss_B, loc=1)
bz_prior = UnivariatePrior('t', 'Bz', gauss_B, loc=-3, scale=2, df=1)
ne_prior = UnivariatePrior('lognorm', 'ne', u.cm**(-3), loc=1, s=1)
cd_prior = UnivariatePrior('lognorm', 'central_density', u.cm**(-3), loc=-1, s=3)
sr_prior = UnivariatePrior('expon', 'scale_radius', u.kpc, loc=-1)
sh_prior = UnivariatePrior('expon', 'scale_height', u.kpc, loc=2)


full_prior = MultivariatePrior({'Bx': bx_prior, 'By': by_prior, 'Bz': bz_prior, 'ne': ne_prior,
                                'central_density': cd_prior, 'scale_radius': sr_prior, 'scale_height': sh_prior
                                })

true_parameters = full_prior.draw_sample()
print(true_parameters)

b1 = ConstantMagneticField(grid)
b2 = b1.radial_component()

print('b1 ', b1.unit)
print('b2 ', b2.unit)

n1 = ConstantThermalElectron(grid)
n2 = ExponentialThermalElectron(grid)

print('n1 ', n1.unit)
print('n2 ', n2.unit)

n3 = n1 + n2
print('n3 ', n3.unit)

bn1 = b2*n3

print('bn1 ', bn1.unit)

r1 = SimpleIntegrator(grid, 0)

s1 = r1@bn1

print('s1 ', s1.unit)

# print(bn1(parameters))

print(s1(true_parameters))

noise_cov = 2.*s1.unit**2
print(isinstance(noise_cov.value, float))

noise_prior = MultivariatePrior('multivariate_normal', unit=s1.unit, cov=noise_cov.value, n=5*2)

noise = noise_prior.draw_sample()

data = s1(true_parameters) + noise

likelihood1 = TOGOGaussianLogLikelihood(data, noise_cov)
likelihood2 = TOGOGaussianLogLikelihood(data, noise_cov, s1, 'likelihood2')

print(likelihood1.output_param_space)
print(likelihood1.input_param_space)
print(likelihood2(true_parameters))
print((likelihood1@s1)(true_parameters))
