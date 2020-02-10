import numpy as np
from imagine import GeneralField

class DummyField(GeneralField):
    """
    Base class for a dummy Field used for sending parameters and settings to
    specific Simulators rather than computing and storing a physical field

    """
    field_name = 'dummy_base'
    field_type = 'dummy'
    @property
    def data(self):
        # A dummy field should never have data called!
        raise NotImplementedError('A dummy field should have no data!')
    @property
    def field_type(self):
        return None

class MagneticField(GeneralField):
    """
    Base class for magnetic fields within IMAGINE
    """
    field_name = 'magnetic_field_base'

    @property
    def field_type(self):
        return 'magnetic_field'
    @property
    def data_description(self):
        return  ['grid_x','grid_y','grid_z','component (x,y,z)']
    @property
    def data_shape(self):
        return tuple([i for i in self.grid.shape] + [3])


class ThermalElectronDensityField(GeneralField):
    """
    Base class for cosmic ray thermal electron density
    """
    field_name = 'thermal_electrons_base'
    
    @property
    def field_type(self):
        return 'cosmic_ray_electron_density'
    @property
    def data_description(self):
        return  ['grid_x','grid_y','grid_z']
    @property
    def data_shape(self):
        return tuple(self.grid.shape)
    
    
class CosmicRayDistribution(GeneralField):
    """
    Base class for cosmic ray thermal electron density


    Parameters
    ----------
    energy_bins : list or array
        Values of the edges of the energy bins in GeV

    Atributes
    ---------
    energy_bins
        Values of the edges of the energy bins in GeV
    data_shape
        aa
    """
    field_name = 'constant_cre_base'
    field_type = 'cosmic_ray_electron_density'
    def __init__(self, *args, energy_bins=[], **kwargs):
        super().__init__(*args, **kwargs)
        # Sets instance attributes
        self.data_description = ['grid_x','grid_y','grid_z','Energy_bin']
        self.energy_bins = np.array(energy_bins)
        self.energy_bins_units = 'GeV'
        number_of_bins = energy_bins.size
        self.data_shape = [i for i in self.grid.shape[i]] + [number_of_bins]
        self.data_units = 'GeV/(m^2 s Sr)'


