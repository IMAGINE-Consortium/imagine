from imagine.tools.icy_decorator import icy
from imagine import Measurements, Simulations
import numpy as np

class Simulator(object):
    """
    Simulator base class

    New Simulators should be introduced by sub-classing the present class.
    Overriding the method :py:meth:`simulate` to convert a list of fields
    into simulated observables. For more details see
    :ref:`design_components:Simulators` section of the documentation.

    Parameters
    ----------
    measurements : imagine.Measurements
        An observables dictionary containing the set of measurements that will be
        used to prepare the mock observables

    Attributes
    ----------
    grid : imagine.Basegrid
        Grid object where the fields were evaluated (NB if a common grid is not being
        used, this is set to None
    fields : list
        List of Fields
    observables : list
        List of Observable keys
    output_units : astropy.units.Unit
        Output units used in the simulator
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
        """
        Called during initalization to store the relevant information in the
        measurements dictionary

        Parameters
        ----------
        measurements : imagine.Measurements
            An observables dictionary containing the set of measurements that will be
            used to prepare the mock observables
        """
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
        """
        Must be overriden with a list or set of simulated quantities this Simulator produces.
        Example: ['fd', 'sync']
        """
        raise NotImplementedError

    @property
    def required_field_types(self):
        """
        Must be overriden with a list or set of required field types Simulator needs.
        Example: ['magnetic_field', 'cosmic_ray_electron_density']
        """
        raise NotImplementedError
    @property
    def allowed_grid_types(self):
        """
        Must be overriden with a list or set of allowed grid types that work with this Simulator.
        Example: ['cartesian']
        """
        raise NotImplementedError

    def simulate(self, key, coords_dict, Nside, output_units):
        """
        Must be overriden with a function that returns the observable described by `key` using
        the fields in self.fields, in units `output_units`.

        Parameters
        ----------
        key : tuple
            Observable key in the standard form ``(data-name,str(data-freq),str(data-Nside)/"tab",str(ext))``
        coords_dict : dictionary
            Dictionary containing coordinates associated with the observable (or None for HEALPix datasets).
        Nside : int
            HEALPix Nside parameter for HEALPix datasets (or None for tabular datasets).
        output_units : astropy.units.Unit
            The physical units that should be used for this mock observable

        Returns
        -------
        numpy.ndarray
            1D *pure* numpy array of length compatible with Nside or coords_dict containing the mock observable
            in the output_units.
        """
        raise NotImplementedError

    def __call__(self, field_list):
        """
        Runs the simulator over a Fields list

        Parameters
        ----------
        field_list : list
            List of imagine.Field object which should include all the `required_field_types`

        Returns
        -------
        sims : imagine.Simulations
            A Simulations object containing all the specified mock data
        """
        self.register_fields(field_list)
        sims = Simulations()
        for key in self.observables:
            sim = self.simulate(key=key, coords_dict=self.output_coords[key],
                                Nside=None, output_units=self.output_units[key])
            sims.append(key, sim[np.newaxis,:], plain=True)
        return sims
