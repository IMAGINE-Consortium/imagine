"""
interface to hammurabiX python wrapper

hammurabi python wrapper hosts a XML tree
where all parameters are registered
the major purpose of this interface
is to do modifications to this XML tree

to accommodate updates of Hampyx in future
only register_observables/fields need modifications
"""

import numpy as np

from imagine.simulators.simulator import Simulator
from imagine.fields.field import GeneralField
from imagine.observables.observable_dict import Measurements, Simulations
from imagine.tools.random_seed import seed_generator
from imagine.tools.icy_decorator import icy
from imagine.tools.timer import Timer
from .hampyx import Hampyx


@icy
class Hammurabi(Simulator):

    def __init__(self, measurements,
                 xml_path='./params.xml',
                 exe_path=None):
        """
        upon initialization, a Hampyx object is initialized
        and its XML tree should be modified according to measurements
        without changing its base file
        :param measurements: Measurements object
        :param exe_path: hammurabi executable path
        :param xml_path: hammurabi xml parameter file path
        """
        self.exe_path = exe_path
        self.xml_path = xml_path
        self.output_checklist = measurements
        self._ham = Hampyx(self._xml_path, self._exe_path)
        self.register_observables()
        self.ensemble_size = int(0)

    @property
    def exe_path(self):
        return self._exe_path

    @exe_path.setter
    def exe_path(self, exe_path):
        self._exe_path = exe_path

    @property
    def xml_path(self):
        return self._xml_path

    @xml_path.setter
    def xml_path(self, xml_path):
        self._xml_path = xml_path

    @property
    def output_checklist(self):
        return self._output_checklist

    @output_checklist.setter
    def output_checklist(self, measurements):
        assert isinstance(measurements, Measurements)
        self._output_checklist = tuple(measurements.keys())

    @property
    def ensemble_size(self):
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        assert isinstance(ensemble_size, int)
        self._ensemble_size = ensemble_size

    def register_observables(self):
        """
        modify hammurabi XML tree according to known output_checklist
        :return:
        """
        # clean up
        try:
            self._ham.del_par(['observable', 'sync'], 'all')
            self._ham.del_par(['observable', 'dm'], 'all')
            self._ham.del_par(['observable', 'faraday'], 'all')
        except ValueError:  # incase no entry in template xml file
            pass
        # refill
        sync_name_cache = list()
        for key in self._output_checklist:
            name, freq, nside, flag = key
            if name == 'sync':
                if (freq, nside) not in sync_name_cache:  # avoid duplication
                    self._ham.add_par(['observable'], 'sync', {'cue': str(1), 'freq': freq, 'nside': nside})
                    sync_name_cache.append((freq, nside))
            elif name == 'fd':
                self._ham.add_par(['observable'], 'faraday', {'cue': str(1), 'nside': nside})
            elif name == 'dm':
                self._ham.add_par(['observable'], 'dm', {'cue': str(1), 'nside': nside})
            else:
                raise ValueError('unrecognised name %s' % name)
        self._ham.print_par(['observable'])

    def register_fields(self, field_list):
        """
        update hammurabi XML tree according to field list
        :param field_list:
        :return:
        """
        for field in field_list:
            # update logical parameters
            controllist = field.field_controllist
            for key, clue in controllist.items():
                assert (len(clue) == 2)
                self._ham.mod_par(clue[0], clue[1])
            # update physical parameters
            checklist = field.field_checklist
            paramlist = field.parameters
            for key, clue in checklist.items():
                assert (len(clue) == 2)
                self._ham.mod_par(clue[0], {clue[1]: str(paramlist[key])})
            # update ensemble size
            self.ensemble_size = field.ensemble_size

    def __call__(self, field_list):
        """
        run hammurabi executable
        pack up outputs in IMAGINE convention
        :param field_list: list of GeneralField objects
        :return: Simulations object
        """
        #t = Timer()
        #t.tick('simulator')
        # update parameters
        self.register_fields(field_list)
        # execute hammurabi ensemble
        sims = Simulations()
        for i in range(self._ensemble_size):
            #t.tick('hamX')
            self._ham()
            #t.tock('hamX')
            # pack up outputs
            for key in self._output_checklist:
                sims.append(key, np.vstack([self._ham.sim_map[key]]))
        # return
        #t.tock('simulator')
        #print(str(t.record))
        return sims
