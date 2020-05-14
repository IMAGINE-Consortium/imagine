from imagine.tools.icy_decorator import icy
from imagine import Measurements, Simulations
import numpy as np

class Simulator(object):
    """
    Simulator base class

    New Simulators must be introduced by sub-classing the present class.
    Overriding the method :py:meth:`simulate` to convert a list of fields
    into simulated observables. For more details see
    :ref:`components:Simulators` section of the documentation.

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
        self.grids = None
        self.use_common_grid = True
        self.fields = None
        self.field_checklist = {}
        self.controllist = {}
        self.observables = []
        self.output_coords = {}
        self.output_units = {}
        self._ensemble_size = None
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


    @property
    def ensemble_size(self):
        return self._ensemble_size

    def register_ensemble_size(self, field_list):
        """
        Checks whether fields have consistent ensemble size and stores this information
        """
        set_of_field_sizes = {f.ensemble_size for f in field_list}
        assert len(set_of_field_sizes)==1, 'All fields should have the same ensemble size'
        self._ensemble_size = set_of_field_sizes.pop()

    def prepare_fields(self, field_list, i):
        """
        Registers the available fields checking whether all requirements
        are satisfied. all data is saved on a dictionary, simulator.fields,
        where field_types are keys.

        The `fields` dictionary is reconstructed for *each realisation* of the
        ensemble. It relies on caching within the Field objects to avoid
        computing the same quantities multiple times.

        If there is more than one field of the same type, they are summed together.

        Parameters
        ----------
        field_list : list
            List containing Field objects
        i : int
            Index of the realisation of the fields that is being registred
        """
        self.fields = {}; self.grid = None

        for field in field_list:
            if field.field_type in (list(self.required_field_types)
                                    + list(self.optional_field_types)):
                # Checks whether the grid_type is correct
                if ((field.grid is not None) and
                    (self.allowed_grid_types is not None)):
                    assert field.grid.grid_type in self.allowed_grid_types, 'Grid type not allowed'

                # Checks whether the grids are consistent
                # (if fields were evaluated on the same grid)
                if self.use_common_grid and (field.field_type != 'dummy'):
                    if self.grid is None:
                        self.grid = field.grid
                    assert self.grid is field.grid, 'Multiple grids when a common grid is required'
                else:
                    if self.grids is None:
                        self.grids = {}
                    elif field.field_type in self.grids:
                        assert self.grids[field.field_type] is field.grid, 'Fields of the same type must have the same grid'
                    else:
                        self.grids[field.field_type] = field.grid

                # Finally, stores the field
                if field.field_type not in self.fields:
                    # Stores the data
                    self.fields[field.field_type] = field._get_data(i)
                    # Stores the checklist dictionary
                    self.field_checklist[field.field_type] = field.field_checklist

                elif field.field_type != 'dummy':
                    # If multiple fields of the same type are present, sums them up
                    self.fields[field.field_type] = (self.fields[field.field_type]
                                                     + field._get_data(i) )
                    # NB the '+=' has *not* been used to changes in the original data
                    # due to its 'inplace' nature
                else:
                    # For multiple dummies, parameters provided by _get_data are
                    # combined (taking care to avoid modifying the original object)
                    self.fields[field.field_type] = self.fields[field.field_type].copy()
                    self.fields[field.field_type].update(field._get_data(i))

                    # The checklists are also combined
                    self.field_checklist[field.field_type] = self.field_checklist[field.field_type].copy()
                    self.field_checklist[field.field_type].update(field.field_checklist)

                if field.field_type == 'dummy':
                    self.controllist[field.field_name] = field.simulator_controllist

        # Makes sure all required fields were included
        assert set(self.required_field_types) <= set(self.fields.keys()), 'Missing required field'

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
        Must be overriden with a list or set of required field types that
        the Simulator needs.
        Example: ['magnetic_field', 'cosmic_ray_electron_density']
        """
        raise NotImplementedError

    @property
    def optional_field_types(self):
        """
        Can be overriden with a list or set of field types that Simulator can use
        if available.
        Example: ['magnetic_field', 'cosmic_ray_electron_density']
        """
        return []

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
            List of imagine.Field object which must include all the `required_field_types`

        Returns
        -------
        sims : imagine.Simulations
            A Simulations object containing all the specified mock data
        """
        sims = Simulations()
        self.register_ensemble_size(field_list)
        for i in range(self.ensemble_size):
            # Prepares all fields
            self.prepare_fields(field_list, i)
            for key in self.observables:
                sim = self.simulate(key=key, coords_dict=self.output_coords[key],
                                    realization_id=i,
                                    output_units=self.output_units[key])
                sims.append(key, sim[np.newaxis,:].to(self.output_units[key]))
        return sims
