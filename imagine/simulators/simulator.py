from imagine.tools.icy_decorator import icy

@icy
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
         
    def register_observables(measurements):
        
        assert isinstance(measurements, Measurements)
        
        for key in measurements:
            if key[0] in simulated_quantities:
                # Using a list for keys as preserving the _order_ may,
                # perhaps, be useful later
                self.observables.append(key) 
                self.output_coords[key] = measurements[key].coords
                self.output_units[key] = measurements[key].unit
        assert len(self.observables)>0, 'No valid observable was requested!'
        
    def register_fields(field_list):
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
    def simulated_quantities():
        raise NotImplementedError
    
    @property
    def required_field_types():
        raise NotImplementedError
    @property
    def allowed_grid_types():
        raise NotImplementedError
    
    def simulate():
        raise NotImplementedError
    def __call__(self, field_list):
        self._register_fields(field_list)
        # The "meat" should be in simulate(), which is called here
        
        raise NotImplementedError

        
        
class Faraday_Simulator(object):
    @property
    def simulated_quantities():
        return {'fd'}
    
    @property
    def required_field_types():
        return {'magnetic_field', 'thermal_electron_density'}
    @property
    def allowed_grid_types():
        return {'cartesian'}