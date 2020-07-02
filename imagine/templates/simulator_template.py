from imagine.simulators import Simulator
import numpy as np
import MY_SIMULATOR

class SimulatorTemplate(Simulator):
    """
    Detailed description of the simulator
    """

    # Class attributes
    SIMULATED_QUANITIES = ['my_observable_quantity']

    # A list or set of what is required for the simulator to work
    REQUIRED_FIELD_TYPES = ['field_type_1', 'field_type_2']

    # Fields which may be used if available
    OPTIONAL_FIELD_TYPES = ['optional_field_type_1']
    ALLOWED_GRID_TYPES = ['grid_type']

    def __init__(self, measurements, extra_args):
        # Send the measurements to parent class
        super().__init__(measurements)

        # Any initialization task involving extra_args
        ...

    def simulate(self, key, coords_dict, realization_id, output_units):
        """
        This is the main function you need to override to create your simulator.
        The simulator will cycle through a series of Measurements and create
        mock data using this `simulate` function for each of them.

        Parameters
        ----------
        key : tuple
            Information about the observable one is trying to simulate
        coords_dict : dictionary
            If the trying to simulate data associated with discrete positions
            in the sky, this dictionary contains arrays of coordinates.
        realization_id : int
            The index associated with the present realisation being computed.
        output_units : astropy.units.Unit
            The requested output units.
        """
        # The argument key provide extra information about the specific
        # measurement one is trying to simulate
        obs_quantity, freq_Ghz , Nside , tag = key

        # If the simulator is working on tabular data, the observed
        # coordinates can be accesd from coords_dict, e.g.
        lat, lon = coords_dict['lat'], coords_dict['lon']

        # Fields can be accessed from a dictionary stored in self.fields
        my_field = self.fields['field_type_1']
        # If a dummy field is being used, instead of an actual realisation,
        # the parameters can be accessed from self.fields['dummy']
        my_dummy_field_parameters = self.fields['dummy']
        # If a common grid is used, it can be accessed from
        grid = self.grid
        # If fields are allowed to use different grids,
        # one can get the grid from
        grid_my_field = my_field.grid

        # SIMULATE
        results = MY_SIMULATOR.simulate(args)
        # The results should be in a 1-D array of size compatible with
        # your dataset. I.e. for tabular data: results.size = lat.size
        # (or any other coordinate)
        # and for HEALPix data  results.size = 12*(Nside**2)

        # Note: Awareness of other observables
        # While this method will be called for each individual observable
        # the other observables can be accessed from self.observables
        # Thus, if your simulator is capable of computing multiple observables
        # at the same time, the results can be saved to an attribute on the first
        # call of `simulate` and accessed from this cache later.
        # To break the degeneracy between multiple realisations (which will
        # request the same key), the realisation_id can be used

        return results
