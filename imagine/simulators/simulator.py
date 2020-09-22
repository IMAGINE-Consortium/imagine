# %% IMPORTS
# Built-in imports
import abc

# Package imports
import numpy as np

# IMAGINE imports
from imagine.observables import Measurements, Simulations
from imagine.tools import BaseClass, req_attr

# All declaration
__all__ = ['Simulator']


# %% CLASS DEFINITIONS
class Simulator(BaseClass, metaclass=abc.ABCMeta):
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
    grids: imagine.Basegrid
        Grid objects for each individual field None if common grid is being used)
    fields : dict
        Dictionary containing field types as keys and the sum of evaluated fields
        as values
    observables : list
        List of Observable keys
    output_units : astropy.units.Unit
        Output units used in the simulator
    """
    def __init__(self, measurements):
        # Call super constructor
        super().__init__()

        self.grid = None
        self.grids = None
        self.fields = None
        self.field_checklist = {}
        self.field_parameters = {}
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
        assert len(self.observables) > 0, 'No valid observable was requested!'

    @property
    def ensemble_size(self):
        return self._ensemble_size

    def register_ensemble_size(self, field_list):
        """
        Checks whether fields have consistent ensemble size and stores this information
        """
        set_of_field_sizes = {f.ensemble_size for f in field_list}
        assert len(set_of_field_sizes) == 1, 'All fields should have the same ensemble size'
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
        self.fields = {}
        self.grid = None
        self.grids = None
        self.field_checklist = {}
        self.controllist = {}

        sorted_field_list = self._sort_field_dependencies(field_list)

        for field in sorted_field_list:
            if field.type in (*self.required_field_types,
                              *self.optional_field_types):

                # Checks whether the grid_type is correct
                if ((field.grid is not None) and
                   (self.allowed_grid_types is not None)):
                    assert field.grid.grid_type in self.allowed_grid_types, 'Grid type not allowed'

                # Checks whether the grids are consistent
                # (if fields were evaluated on the same grid)
                if self.use_common_grid and (field.type != 'dummy'):
                    if self.grid is None:
                        self.grid = field.grid
                    assert self.grid is field.grid, 'Multiple grids when a common grid is required'
                else:
                    if self.grids is None:
                        self.grids = {}
                    elif field.type in self.grids:
                        assert self.grids[field.type] is field.grid, 'Fields of the same type must have the same grid'
                    else:
                        self.grids[field.type] = field.grid

                # Organises dependencies
                dependencies = {}
                for dep in field.dependencies_list:
                    if isinstance(dep, str):
                        # If a string is used, dep is actually a field_type
                        dependencies[dep] = self.fields[dep]
                    else:
                        # Otherwise, dep corresponds to a class
                        for other_field in sorted_field_list:
                            if other_field is field:
                                continue
                            # Stores the requested object in the dictionary
                            if isinstance(other_field, dep):
                                dependencies[dep] = other_field
                                break

                # Finally, stores the field
                if field.type not in self.fields:
                    # Stores the data
                    self.fields[field.type] = field.get_data(i, dependencies)
                    # Stores the parameters list
                    self.field_parameters[field.type] = field.parameter_names
                    # Stores the checklist (for dummies only)
                    if field.type == 'dummy':
                        self.field_checklist = field.field_checklist.copy()

                elif field.type != 'dummy':
                    # If multiple fields of the same type are present, sums them up
                    self.fields[field.type] = (self.fields[field.type]
                                                     + field.get_data(i, dependencies))
                    # NB the '+=' has *not* been used to changes in the original data
                    # due to its 'inplace' nature
                else:
                    # For multiple dummies, parameters provided by _get_data are
                    # combined (taking care to avoid modifying the original object)
                    self.fields[field.type] = self.fields[field.type].copy()
                    self.fields[field.type].update(field.get_data(i, dependencies))

                    # The checklists are also combined
                    self.field_checklist.update(field.field_checklist)

                if field.type == 'dummy':
                    self.controllist[field.name] = field.simulator_controllist

        # Makes sure all required fields were included
        assert set(self.required_field_types) <= set(self.fields.keys()), 'Missing required field'

    def _sort_field_dependencies(self, fields):
        """
        Reorders a fields list so that dependencies are evaluated before
        the dependent fields.

        Parameters
        ----------
        fields : list
            List of Fields which may contain dependencies

        Returns
        -------
        sorted_fields : list
            List of sorted Fields
        """
        independent_fields, dependencies = self._find_dependencies(fields)
        sorted_fields = self._solve_dependencies(independent_fields, dependencies)
        return sorted_fields

    def _find_dependencies(self, fields):
        """
        Reads a list of Fields and constructs a list of independent fields and dictionary
        containing all field depenencies. Dependencies on 'field_type' are converted to
        dependencies on classes.

        Parameters
        ----------
        fields : list
            Initial list of fields

        Returns
        -------
        independent_fields_list : list
            List containing fields with no dependencies
        dependencies : dict
            Dictionary containing field objects as keys and the classes they depend on
            as values.
        """
        field_types = {}
        dependencies = {}
        independent_fields_list = []

        # Prepares field_type and dependencies dictionaries
        for field in fields:
            ftype = field.type
            fclass = type(field)
            fdep = field.dependencies_list

            if ftype not in field_types:
                field_types[ftype] = {fclass}
            else:
                field_types[ftype].add(fclass)

            if len(fdep) == 0:
                independent_fields_list.append(field)
            else:
                if field not in dependencies:
                    dependencies[field] = set(fdep)
                else:
                    dependencies[field].update(fdep)

        # Subsititutes any field type string by field classes
        for deps in dependencies.values():
            for dep in set(deps):
                if isinstance(dep, str):
                    deps.remove(dep)
                    deps.update(field_types[dep])

        return independent_fields_list, dependencies

    def _solve_dependencies(self, independent_fields, dependencies,
                            max_iter=100, overwrite=True):
        """
        Applied basic topological sorting to the field dependenceis

        Parameters
        ----------
        independent_fields : list
            List of field objects with no dependencies
        dependencies : dict
            Dictionary containing field objects as keys and the classes they depend on
            as values.
        max_iter : int, optional
            Maximum number of iterations while trying to solve the dependencies
        overwrite : bool
            If True (default),`independent_fields` and `dependencies` will be modyfied
            by this method

        Returns
        -------
        L : list
            Sorted list of field object
        """
        L = []  # Empty list that will contain the sorted elements
        if overwrite:
            S = independent_fields  # Set of all nodes with no incoming edge
            deps = dependencies
        else:
            from copy import deepcopy
            S = independent_fields.copy()
            deps = deepcopy(dependencies)

        counter = 0
        while S:
            counter += 1
            assert counter < max_iter, 'Error: too many iterations'

            # Removes a node n from S
            n = S.pop()

            # Appends n to tail of L
            L.append(n)

            # Goes through all the nodes
            for m in list(deps.keys()):
                edges = deps[m]
                # If n is in the edges, removes it
                if type(n) in edges:
                    edges.remove(type(n))
                # If there are no edges, add it to the
                # independent nodes list
                if not edges:
                    S.append(m)
                    del deps[m]

        assert not deps, 'There is a cyclical Field dependency!'

        return L

    @property
    @req_attr
    def simulated_quantities(self):
        """
        Must be overriden with a list or set of simulated quantities this Simulator produces.
        Example: ['fd', 'sync']
        """
        return(self.SIMULATED_QUANTITIES)

    @property
    @req_attr
    def required_field_types(self):
        """
        Must be overriden with a list or set of required field types that
        the Simulator needs.
        Example: ['magnetic_field', 'cosmic_ray_electron_density']
        """
        return(self.REQUIRED_FIELD_TYPES)

    @property
    def optional_field_types(self):
        """
        Can be overriden with a list or set of field types that Simulator can use
        if available.
        Example: ['magnetic_field', 'cosmic_ray_electron_density']
        """
        return(getattr(self, 'OPTIONAL_FIELD_TYPES', []))

    @property
    @req_attr
    def allowed_grid_types(self):
        """
        Must be overriden with a list or set of allowed grid types that work with this Simulator.
        Example: ['cartesian']
        """
        return self.ALLOWED_GRID_TYPES

    @property
    def use_common_grid(self):
        """
        Must be overriden with a list or set of allowed grid types that work with this Simulator.
        Example: ['cartesian']
        """
        return getattr(self, 'USE_COMMON_GRID', True)

    @abc.abstractmethod
    def simulate(self, key, coords_dict, realization_id, output_units):
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
        for i in range(self._ensemble_size):
            # Prepares all fields
            self.prepare_fields(field_list, i)
            for key in self.observables:
                sim = self.simulate(key=key, coords_dict=self.output_coords[key],
                                    realization_id=i,
                                    output_units=self.output_units[key])
                sims.append(name=key,
                            data=sim[np.newaxis, :].to(self.output_units[key]),
                            coords=self.output_coords[key])
        return sims
