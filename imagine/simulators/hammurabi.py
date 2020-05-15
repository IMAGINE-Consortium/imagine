import numpy as np
import logging as log
from imagine.simulators.simulator import Simulator
from imagine.tools.icy_decorator import icy
from imagine.tools.timer import Timer
from hampyx import Hampyx
import astropy.units as u



@icy
class Hammurabi(Simulator):
    """
    This is an interface to hammurabi X Python wrapper.

    Upon initialization, a Hampyx object is initialized
    and its XML tree should be modified according to measurements
    without changing its base file.

    Parameters
    ----------
    measurements : Measurements object
        IMAGINE defined dictionary of measured data.

    exe_path : string
        Absolute hammurabi executable path.

    xml_path : string
        Absolute hammurabi xml parameter file path.
    """
    def __init__(self, measurements, xml_path, exe_path=None,
                 ):
        log.debug('@ hammurabi::__init__')
        super().__init__(measurements)
        self.exe_path = exe_path
        self.xml_path = xml_path
        self.current_realization = -1
        # Initializes Hampyx
        self._ham = Hampyx(self.xml_path, self.exe_path)
        # Makes the modifications required by measurements to the XML file
        self.initialize_ham_xml()

    @property
    def simulated_quantities(self):
        return {'fd','dm', 'sync'}
    @property
    def required_field_types(self):
        return ['dummy']
    @property
    def optional_field_types(self):
        return ['magnetic_field', 'thermal_electron_density' ,
                'cosmic_ray_electron_density']
    @property
    def allowed_grid_types(self):
        return {'cartesian'}

    def initialize_ham_xml(self):
        """
        Modify hammurabi XML tree according to the requested measurements.
        """
        log.debug('@ hammurabi::initialize_ham_xml')
        # Cleans up previously defined entries
        for t in ('sync','dm','faraday'):
            try:
                self._ham.del_par(['observable', t], 'all')
            except ValueError:  # in case no entry in template xml file
                pass

        # Includes the new data
        sync_name_cache = list()
        for key in self.observables:
            name, freq, nside, flag = key

            if nside=='tab':
                raise NotImplementedError('Tabular datasets not yet supported!')

            if name == 'sync':
                if (freq, nside) not in sync_name_cache:  # Avoids duplication
                    self._ham.add_par(['observable'], 'sync',
                                      {'cue': str(1), 'freq': freq, 'nside': nside})
                    sync_name_cache.append((freq, nside))
            elif name == 'fd':
                self._ham.add_par(['observable'], 'faraday',
                                  {'cue': str(1), 'nside': nside})
            elif name == 'dm':
                self._ham.add_par(['observable'], 'dm',
                                  {'cue': str(1), 'nside': nside})
            else:
                raise ValueError('unrecognised name %s' % name)
        self._ham.print_par(['observable'])

    def update_hammurabi_settings(self):
        """
        Updates hamx XML tree using the provided controllist

        This is used to configure logical parameters (e.g. enable or disable
        a given hamx module).
        """
        # This replaces the old `register_fields` method
        log.debug('@ hammurabi::update_hammurabi_settings')

        for controllist in self.controllist.values():
            # The field names (the dictionary keys) are unimportant
            for keychain, attrib in controllist.values():
                # The keys in each hamx controllist are also unimportant
                self._ham.mod_par(keychain, attrib)

    def update_hammurabi_parameters(self):
        """
        Updates hammurabi XML tree according to field checklists and parameter
        choices

        This is used to configure physical parameters
        """
        # This replaces the old `update_fields` method
        log.debug('@ hammurabi::update_hammurabi_parameters')

        checklist = self.field_checklist['dummy']
        parameters = self.fields['dummy']

        # hammurabiX does not support int64 seeds
        parameters['random_seed'] = np.uint16(parameters['random_seed'])

        for name, (keychain, attribute_tag) in checklist.items():
            self._ham.mod_par(keychain, {attribute_tag: str(parameters[name])})

    def simulate(self, key, coords_dict, realization_id, output_units):

        # If the realization_id is different from the self.current_realization
        # hammurabi needs to re-run to (re)generate the observables
        if (self.current_realization != realization_id):
            self.update_hammurabi_settings()
            # update parameters
            self.update_hammurabi_parameters()
            # Runs Hammurabi
            self._ham()

        # Adjusts the units (without copy) and returns
        return self._ham.sim_map[key] << self._units(key)

    def _units(self, key):
        if key[0] == 'sync':
            if key[3] in ('I','Q','U','PI'):
                return u.K
            elif key[3] == 'PA':
                return u.rad
            else:
                raise ValueError
        elif key[0] == 'fd':
            return u.rad/u.m/u.m
        elif key[0] == 'dm':
            return u.pc/u.cm**3
        else:
            raise Value



