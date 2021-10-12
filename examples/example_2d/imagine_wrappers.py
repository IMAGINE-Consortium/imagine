import imagine as img
import nifty7 as ift
import astropy as apy
import astropy.units as u
import numpy as np

from functions_nifty import get_spiral_model, get_random_model, get_mock_truth


class SpiralField(img.fields.Field):
    """ Here comes the description of the electron density model """

    def __init__(self, grid, parameters, ensemble_size, ensemble_seeds):
        super().__init__(grid)
        self.ensemble_size = ensemble_size
        self.ensemble_seeds = ensemble_seeds
        self.parameters = parameters
        self.n_data = grid.x.shape[0]
        self.n_depth = grid.x.shape[1]
        self.domain = ift.makeDomain(ift.RGSpace((self.n_data, self.n_depth,)),)
        self.unit = ''

    # Class attributes
    NAME = 'spiral'

    UNITS = u.Unit('')
    # Is this field stochastic or not. Only necessary if True
    STOCHASTIC_FIELD = False
    # If there are any dependencies, they should be included in this list
    DEPENDENCIES_LIST = []
    # List of all parameters for the field
    PARAMETER_NAMES = ['log_a']

    TYPE = 'spiral'

    @property
    def data_description(self):
        return(['grid_x', 'grid_y', 'grid_z'])

    @property
    def data_shape(self):
        return tuple(self.grid.shape)

    def compute_field(self, seed):
        param_domain = ift.makeDomain(ift.DomainTuple.scalar_domain())
        if isinstance(self.parameters['log_a'], u.quantity.Quantity):
            p = self.parameters['log_a'].to_value()
        else:
            p = self.parameters['log_a']
        a = ift.MultiField.from_dict({'log_a': ift.Field(param_domain, p)})
        spiral_m = get_spiral_model(self.domain, {})
        return spiral_m(a).val[..., np.newaxis] << u.Unit(self.unit)


class SpiralFieldFactory(img.fields.FieldFactory):

    """"""
    def __init__(self, grid):
        super().__init__(grid=grid)
    # Class attributes
    # Field class this factory uses
    FIELD_CLASS = SpiralField

    # Default values are used for inactive parameters
    DEFAULT_PARAMETERS = {'log_a': 1., }

    # All parameters need a range and a prior
    PRIORS = {'log_a': img.priors.GaussianPrior(mu=0, sigma=1),
              }


class RandomField(img.fields.Field):
    """ Here comes the description of the electron density model """

    def __init__(self, grid, parameters, ensemble_size, ensemble_seeds):
        super().__init__(grid)
        self.ensemble_size = ensemble_size
        self.ensemble_seeds = ensemble_seeds
        self.parameters = parameters
        self.n_data = grid.x.shape[0]
        self.n_depth = grid.x.shape[1]
        self.domain = ift.makeDomain((ift.RGSpace((self.n_data, self.n_depth,)),))
        self.unit = ''

    # Class attributes
    NAME = 'random'
    UNITS = u.Unit('')
    # Is this field stochastic or not. Only necessary if True
    STOCHASTIC_FIELD = True
    # If there are any dependencies, they should be included in this list
    DEPENDENCIES_LIST = []
    # List of all parameters for the field
    PARAMETER_NAMES = ['a0', 'k0', 'p']

    TYPE = 'random'

    @property
    def data_description(self):
        return(['grid_x', 'grid_y', 'grid_z'])

    @property
    def data_shape(self):
        return tuple(self.grid.shape)

    def compute_field(self, seed):
        ift.random.push_sseq_from_seed(int(seed))
        params = {}
        for k, p in self.parameters.items():
            if isinstance(p, u.quantity.Quantity):
                params.update({k: p.to_value()})
            else:
                params.update({k: p})
        random_m = get_random_model(self.domain, params)
        a = ift.from_random(random_m.domain)
        return random_m(a).val[..., np.newaxis] << u.Unit(self.unit)


class RandomFieldFactory(img.fields.FieldFactory):
    """"""

    def __init__(self, domain):
        grid = domain
        super().__init__(grid=grid)
    # Class attributes
    # Field class this factory uses
    FIELD_CLASS = RandomField

    # Default values are used for inactive parameters
    DEFAULT_PARAMETERS = {'a0': 0.01, 'k0': 3, 'p': -3}

    # All parameters need a range and a prior
    PRIORS = {'p': img.priors.GaussianPrior(mu=-3, sigma=1),
              'a0': img.priors.FlatPrior(xmin=0.001, xmax=10),
              'k0': img.priors.FlatPrior(xmin=1, xmax=10),
              }


class MockSpiralSimulator(img.simulators.Simulator):
    """
    Example simulator to illustrate
    """

    # Class attributes
    SIMULATED_QUANTITIES = ['test']
    REQUIRED_FIELD_TYPES = ['spiral']
    OPTIONAL_FIELD_TYPES = ['random']
    ALLOWED_GRID_TYPES = ['cartesian']

    def __init__(self, measurements, grid, response):
        # Send the Measurements to the parent class
        super().__init__(measurements)
        self.grid = grid
        self.n_data = grid.x.shape[0]
        self.n_depth = grid.x.shape[1]
        self.domain = ift.makeDomain(ift.RGSpace((self.n_data, self.n_depth,)), )
        self.response = response
        self.unit = u.Unit('')
        # Stores class-specific attributes

    def simulate(self, key, coords_dict, realization_id, output_units):
        # Accesses fields and grid
        field_list = []
        if 'random' in self.fields:
            field_list.append(ift.Field(self.domain, self.fields['random'].to_value()[:, :, 0]))
        field_list.append(ift.Field(self.domain, self.fields['spiral'].to_value()[:, :, 0]))
        signal_response = self.response(ift.utilities.my_sum(field_list))
        return signal_response.val << self.unit


def make_observable_dict(response, noise_sigma, model_dict, truth_dicts):
    position = {}
    true_fields = {}
    nifty_data = {}

    for key, model in model_dict.items():
        true_realization, latent_position = get_mock_truth(model, truth_dicts[key])
        position.update(latent_position)
        true_fields.update({key: true_realization})

    true_signal_response = response(ift.utilities.my_sum(true_fields.values()))
    N = ift.ScalingOperator(true_signal_response.domain, noise_sigma ** 2)
    data = true_signal_response + N.draw_sample_with_dtype(dtype=float)
    nifty_data.update({'sres': true_signal_response, 'data': data, 'N': N,
                  'position': ift.MultiField.from_dict(position)})
    data = apy.table.Table({'meas': data.val,
                            'err': np.ones_like(data.val) * noise_sigma,
                            })
    mock_dataset = img.observables.TabularDataset(data, name='test', data_col='meas', err_col='err')

    return img.observables.Measurements(mock_dataset), nifty_data, true_fields
