from imagine import MagneticField, ThermalElectronDensityField
from imagine import GeneralFieldFactory, FlatPrior
from imagine.tools.icy_decorator import icy
import numpy as np
import astropy.units as u
import scipy.stats as stats

@icy
class CosThermalElectronDensity(ThermalElectronDensityField):
    r"""
    Toy model for naively oscilating thermal electron distribution following:

    .. math::

        n_e(x,y,z) = n_0 [1+\cos (a x + \alpha)][1+\cos (b y + \beta)][1+\cos(c y + \gamma)]

    The field parameters are: 'n0', which corresponds to :math:`n_0`; and
    'a', 'b', 'c', 'alpha', 'beta', 'gamma', which are
    :math:`a`, :math:`b`, :math:`c`, :math:`\alpha`, :math:`\beta`, :math:`\gamma`, respectively.
    """
    field_name = 'cos_therm_electrons'
    stochastic_field = False

    @property
    def field_checklist(self):
        return {'n0' : None, 'a' : None, 'alpha' : None,
                'b' : None, 'beta' : None, 'c' : None, 'gamma' : None}

    def compute_field(self, seed):
        x = self.grid.x
        y = self.grid.y
        z = self.grid.z
        p = self.parameters

        return ( p['n0']*(1 + np.cos(p['a']*x + p['alpha']))
                        *(1 + np.cos(p['b']*y + p['beta']))
                        *(1 + np.cos(p['c']*z + p['gamma'])) )


class CosThermalElectronDensity_Factory(GeneralFieldFactory):
    """
    Field factory associated with the :py:class:`CosThermalElectronDensityw`
    class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_class = CosThermalElectronDensity
        self.default_parameters = {'n0': 1*u.cm**-3,
                                   'a': 0.0/u.kpc*u.rad,
                                   'b': 0.0/u.kpc*u.rad,
                                   'c': 0.0/u.kpc*u.rad,
                                   'alpha': 0.*u.rad,
                                   'beta':  0.*u.rad,
                                   'gamma': 0.*u.rad}
        k = FlatPrior(interval=[0.01,100]*u.rad/u.kpc)
        d = FlatPrior(interval=[0,2*np.pi]*u.rad/u.kpc)
        self.priors = {'n0': FlatPrior(interval=[0,10]*u.cm**-3),
                       'a': k, 'b': k, 'c': k,
                       'alpha': d, 'beta': d, 'gamma': d}

@icy
class NaiveGaussianMagneticField(MagneticField):
    r"""
    Toy model for naive Gaussian random field for testing.

    The values of each of the magnetic field components are individually
    drawn from a Gaussian distribution with mean 'a0' and
    standard deviation 'b0'.

    Warning: divergence may be non-zero!
    """
    field_name = 'naive_gaussian_magnetic_field'
    stochastic_field = True

    @property
    def field_checklist(self):
        return {'a0' : None, 'b0' : None}

    def compute_field(self, seed):

        # Creates an empty array to store the result
        B = np.empty(self.data_shape) * self.field_units

        mu = self.parameters['a0'].to_value(self.field_units)
        sigma = self.parameters['b0'].to_value(self.field_units)

        # Draws values from a normal distribution with these parameters
        # using the seed provided in the argument
        distr = stats.norm(loc=mu, scale=sigma)
        B = distr.rvs(size=self.data_shape, random_state=seed)

        return B*self.field_units


class NaiveGaussianMagneticField_Factory(GeneralFieldFactory):
    """
    Field factory associated with the :py:class:`NaiveGaussianMagneticField`
    class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_class = NaiveGaussianMagneticField
        self.default_parameters = {'a0': 1*u.microgauss,
                                   'b0': 0.1*u.microgauss}

        self.priors = {'a0': FlatPrior(interval=[-20,20]*u.microgauss),
                       'b0': FlatPrior(interval=[-20,20]*u.microgauss)}

