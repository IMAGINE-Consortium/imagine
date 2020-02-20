from imagine.tools.icy_decorator import icy
from imagine import Measurements, Simulations
import numpy as np

class Simulator(object):
    """
    Simulator base class
    
    WORK IN PROGRESS new simulator base class!
    
    Should contain all the book-keeping and checking, with
    the important calculations delegated to the simulate() method and
    parameters in subclass-defined properties 
    """
    def __init__(self, measurements):
        self.grid = None
        self.use_common_grid = True
        self.fields = None
        self.observables = []
        self.output_coords = {}
        self.output_units = {}
        self.register_observables(measurements)
         
    def register_observables(self, measurements):
        
        assert isinstance(measurements, Measurements)
        
        for key in measurements.keys():
            if key[0] in self.simulated_quantities:
                # Using a list for keys as preserving the _order_ may,
                # perhaps, be useful later
                self.observables.append(key) 
                self.output_coords[key] = measurements[key].coords
                self.output_units[key] = measurements[key].unit
        assert len(self.observables)>0, 'No valid observable was requested!'
        
    def register_fields(self, field_list):
        """
        Registers the available fields checking wheter requirements 
        are satisfied. Everything is saved on a dictionary, 
        Faraday_Simulator.fields, where field_types are keys.
        
        Parameters
        ----------
        field_list : list
            List containing Field objects
        """
        self.fields = {}; self.grid = None
        
        for field in field_list:
            if field.field_type in self.required_field_types:
                # Makes sure this is the only field of this type
                assert field.field_type not in self.fields, 'More than one field of the same type'
                
                # Checks whether the grid_type is correct
                if ((field.grid is not None) and 
                    (self.allowed_grid_types is not None)):
                    assert field.grid.grid_type in self.allowed_grid_types, 'Grid type not allowed'
                
                # Checks whether the grids are consistent
                # (if fields were evaluated on the same grid)
                if self.use_common_grid:
                    if self.grid is None:
                        self.grid = field.grid
                    assert self.grid is field.grid, 'Multiple grids when a common grid is required'
                # Finally, stores the field
                self.fields[field.field_type] = field
        
        # Makes sure all required fields were included
        assert set(self.required_field_types) == set(self.fields.keys()), 'Missing required field'
        
    @property
    def simulated_quantities(self):
        raise NotImplementedError
    
    @property
    def required_field_types(self):
        raise NotImplementedError
    @property
    def allowed_grid_types(self):
        raise NotImplementedError
    
    def simulate(self, key, coords_dict, Nside, output_units):
        raise NotImplementedError
    
    def __call__(self, field_list):
        self.register_fields(field_list)
        sims = Simulations()
        for key in self.observables:
            sim = self.simulate(key=key, coords_dict=self.output_coords[key], 
                                Nside=None, output_units=self.output_units[key])
            sims.append(key, sim[np.newaxis,:], plain=True)
        return sims
        