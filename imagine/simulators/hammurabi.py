# %% IMPORTS
# Built-in imports
import logging as log
from os import path
import tempfile

# Package imports
import astropy.units as u
import hampyx
from hampyx import Hampyx
import numpy as np

# IMAGINE imports
import imagine as img
from imagine.simulators import Simulator

# All declaration
__all__ = ['Hammurabi']


# %% CLASS DEFINITIONS
class Hammurabi(Simulator):
    """
    This is an interface to hammurabi X Python wrapper.

    Upon initialization, a Hampyx object is initialized
    and its XML tree should be modified according to measurements
    without changing its base file.

    If a `xml_path` is provided, it will be used as the base XML tree,
    otherwise, hammurabiX's default 'params_template.xml' will be used.

    :py:class:`Dummy` fields can be used to change the parameters of hammurabiX.
    Other fields are temporarily saved to disk (using :py:data:`imagine.rc`
    'temp_dir' directory) and loaded into hammurabi.

    Parameters
    ----------
    measurements : Measurements object
        IMAGINE defined dictionary of measured data.
    exe_path : string
        Absolute hammurabi executable path.
    xml_path : string
        Absolute hammurabi xml parameter file path.
    """

    # Class attributes
    SIMULATED_QUANTITIES = ['fd', 'dm', 'sync']
    REQUIRED_FIELD_TYPES = []
    OPTIONAL_FIELD_TYPES = ['dummy','magnetic_field',
                            'thermal_electron_density',
                            'cosmic_ray_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']


    def __init__(self, measurements, xml_path=None, exe_path=None):
        log.debug('@ hammurabi::__init__')
        super().__init__(measurements)

        if exe_path is not None:
            self.exe_path = exe_path
        else:
            # Uses standard hamx path
            self.exe_path = img.rc['hammurabi_hamx_path']

        self.exe_path = exe_path
        if xml_path is not None:
            self.xml_path = xml_path
        else:
            # Uses standard hammurabi template
            hampydir = path.dirname(hampyx.__file__)
            self.xml_path = path.join(hampydir, '../templates/params_template.xml')

        self.current_realization = -1
        # Initializes Hampyx
        self._ham = Hampyx(self.xml_path, self.exe_path)
        # Sets Hampyx's working directory
        self._ham.wk_dir = img.rc['temp_dir']
        # Makes the modifications required by measurements to the XML file
        self.initialize_ham_xml()
        # List of files containing evaluations of fields
        self._field_dump_files = []

    def initialize_ham_xml(self):
        """
        Modify hammurabi XML tree according to the requested measurements.
        """
        log.debug('@ hammurabi::initialize_ham_xml')
        # Cleans up previously defined entries
        for t in ('sync', 'dm', 'faraday'):
            try:
                self._ham.del_par(['observable', t], 'all')
            except ValueError:  # in case no entry in template xml file
                pass

        # Includes the new data
        sync_name_cache = []
        for key in self.observables:
            name, freq, nside, flag = key

            if nside == 'tab':
                raise NotImplementedError('Tabular datasets not yet supported!')

            if name == 'sync':
                if (freq, nside) not in sync_name_cache:  # Avoids duplication
                    self._ham.add_par(['observable'], 'sync',
                                      {'cue': '1', 'freq': freq, 'nside': nside})
                    sync_name_cache.append((freq, nside))
            elif name == 'fd':
                self._ham.add_par(['observable'], 'faraday',
                                  {'cue': '1', 'nside': nside})
            elif name == 'dm':
                self._ham.add_par(['observable'], 'dm',
                                  {'cue': '1', 'nside': nside})
            else:
                raise ValueError('unrecognised name %s' % name)
        self._ham.print_par(['observable'])

    def _update_hammurabi_settings(self):
        """
        Updates hamx XML tree using the provided controllist

        This is used to configure logical parameters (e.g. enable or disable
        a given hamx module).
        """
        # This replaces the old `register_fields` method
        log.debug('@ hammurabi::_update_hammurabi_settings')

        for controllist in self.controllist.values():
            # The field names (the dictionary keys) are unimportant
            for keychain, attrib in controllist.values():
                # The keys in each hamx controllist are also unimportant
                self._ham.mod_par(keychain, attrib)

    def _update_hammurabi_parameters(self):
        """
        Updates hammurabi XML tree according to field checklists and parameter
        choices

        This is used to configure physical parameters
        """
        # This replaces the old `update_fields` method
        log.debug('@ hammurabi::_update_hammurabi_parameters')
        if 'dummy' not in self.field_checklist:
            return

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
            self._update_hammurabi_settings()
            # Updates parameters
            self._update_hammurabi_parameters()
            # Saves the fields to disk
            self._dump_fields()
            # Runs Hammurabi
            self._ham()
            # Cleans-up
            self._clean_up_dumped_fields()

        # Adjusts the units (without copy) and returns
        return self._ham.sim_map[key] << self._units(key)

    def _units(self, key):
        if key[0] == 'sync':
            if key[3] in ('I', 'Q', 'U', 'PI'):
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
            raise ValueError

    def _dump_fields(self, use_rnd=False):
        """Dumps field data to disk as binnary files"""
        for field_type, field_data in self.fields.items():
            if field_type == 'thermal_electron_density':
                if use_rnd:
                    hamx_key = 'ternd'
                else:
                    hamx_key = 'tereg'
            elif field_type == 'magnetic_field':
                if use_rnd:
                    hamx_key = 'brnd'
                else:
                    hamx_key = 'breg'
            elif field_type == 'cosmic_ray_electron_density':
                hamx_key = 'cre'
            else:
                # For any other field_type: skip
                continue

            # Points to the correct grid
            grid = self.grid if (self.grid is not None) else self.grids[field_type]

            # Adjusts grid parameters
            for coord, n, (cmin, cmax) in zip(['x', 'y', 'z'],
                                              grid.resolution,
                                              grid.box):
                self._ham.mod_par(['grid', 'box_'+hamx_key, 'n'+coord],
                                  {'value': str(n)})
                self._ham.mod_par(['grid', 'box_'+hamx_key, coord+'_min'],
                                  {'value': str(cmin.value)})
                self._ham.mod_par(['grid', 'box_'+hamx_key, coord+'_max'],
                                  {'value': str(cmax.value)})

            # Generates temporary file
            data_dump_file = tempfile.NamedTemporaryFile(prefix=hamx_key+'_',
                                                         suffix='.bin',
                                                         dir=img.rc['temp_dir'])
            # Dumps the data
            (field_data.value).tofile(data_dump_file)

            # Instructs Hammurabi to read the file
            self._ham.mod_par(['fieldio', hamx_key],
                              {'read': '1', 'filename': data_dump_file.name})
            # Appends a reference to files list
            self._field_dump_files.append(data_dump_file)

    def _clean_up_dumped_fields(self):
        """Removes any temporary field data dump files"""
        for f in self._field_dump_files:
            f.close()
        self._field_dump_files.clear()
