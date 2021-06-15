# %% IMPORTS
import nifty7 as ift
# IMAGINE imports
from imagine.fields import DummyField, FieldFactory
from imagine.priors import FlatPrior

from imagine.fields.basic_fields import MagneticField

# All declaration
__all__ = ['BrndES', 'BrndESFactory']


class RandomThermalElectrons(ThermalElectronDensityField):
    """
    Thermal electron densities drawn from a Gaussian distribution

    NB This may lead to negative densities depending on the choice of
    parameters. This may be controlled with the 'min_ne' parameter
    which sets a minimum value for the density field (i.e.
    any value smaller than the minimum density is set to min_ne).

    The field parameters are: 'mean', the mean of the distribution; 'std', the
    standard deviation of the distribution; and 'min_ne', the
    aforementioned minimum density. To disable the minimum density requirement,
    it may be set to NaN.
    """

    # Class attributes
    NAME = 'random_thermal_electrons'
    STOCHASTIC_FIELD = True
    PARAMETER_NAMES = ['mean', 'std', 'min_ne']

    def compute_field(self, seed):
        # Converts dimensional parameters into numerical values
        # in the correct units (stats norm does not like units)
        mu = self.parameters['mean'].to_value(self.units)
        sigma = self.parameters['std'].to_value(self.units)
        minimum_density = self.parameters['min_ne'].to_value(self.units)

        # Draws values from a normal distribution with these parameters
        # using the seed provided in the argument
        distr = stats.norm(loc=mu, scale=sigma)
        result = distr.rvs(size=self.data_shape, random_state=seed)

        # Applies minimum density, if present
        if np.isfinite(minimum_density):
            result[result < minimum_density] = minimum_density

        return result << self.units  # Restores units

# %% CLASS DEFINITIONS
class BrndES(MagneticField):
    """
    This field uses nifty to construct the random magnetic field
    ES random GMF
    """

    # Class attributes
    NAME = 'brnd_ES'
    STOCHASTIC_FIELD = True
    PARAMETER_NAMES = ['offset', 'offset_std', 'fluctuations', 'k1', 'a1', 'rho', 'r0', 'z0']

    def __init__(self, *args, grid, random_seed, **kwargs):
        super().__init__(*args, **kwargs)
        # Default controllist
        self.grid = grid
        self.random_seed = random_seed

    def compute_field(self, seed):
        offset = self.parameters['mean'].to_value(self.units)
        sigma = self.parameters['std'].to_value(self.units)
        minimum_density = self.parameters['min_ne'].to_value(self.units)


class BrndESFactory(FieldFactory):
    """
    Field factory that produces the dummy field :py:class:`BrndES`
    (see its docs for details).
    """

    # Class attributes
    FIELD_CLASS = BrndES
    DEFAULT_PARAMETERS = {'rms': 2,
                          'k0': 10,
                          'a0': 1.7,
                          'k1': 0.1,
                          'a1': 0,
                          'rho': 0.5,
                          'r0': 8,
                          'z0': 1}
    PRIORS = {'rms': FlatPrior(xmin=0, xmax=4),
              'k0': FlatPrior(xmin=0.1, xmax=1),
              'a0': FlatPrior(xmin=1, xmax=3),
              'k1': FlatPrior(xmin=0.01, xmax=1),
              'a1': FlatPrior(xmin=0, xmax=3),
              'rho': FlatPrior(xmin=0, xmax=1),
              'r0': FlatPrior(xmin=2, xmax=10),
              'z0': FlatPrior(xmin=0.1, xmax=3)}

    def __init__(self, *args, grid_nx=None,
                 grid_ny=None, grid_nz=None, **kwargs):
        super().__init__(*args, **kwargs,
                         field_kwargs={'grid_nx': grid_nx,
                                       'grid_ny': grid_ny,
                                       'grid_nz': grid_nz})
