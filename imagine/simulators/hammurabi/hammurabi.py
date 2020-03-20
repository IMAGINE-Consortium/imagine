import numpy as np
import logging as log
from imagine.simulators.simulator import Simulator
from imagine.observables.observable_dict import Measurements, Simulations
from imagine.tools.icy_decorator import icy
from imagine.tools.timer import Timer
from hampyx import Hampyx


@icy
class Hammurabi(Simulator):
    """
    This is an interface to hammurabi X Python wrapper.
    """
    
    def __init__(self, measurements,
                 xml_path=None,
                 exe_path=None):
        """
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
        log.debug('@ hammurabi::__init__')
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
        Modify hammurabi XML tree according to known output_checklist.
        """
        log.debug('@ hammurabi::register_observables')
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
        Update hammurabi XML tree according to field list controllist.
        """
        log.debug('@ hammurabi::register_fields')
        for field in field_list:
            # update logical parameters
            controllist = field.field_controllist
            for key, clue in controllist.items():
                assert (len(clue) == 2)
                self._ham.mod_par(clue[0], clue[1])
            # update ensemble size
            self.ensemble_size = field.ensemble_size

    def update_fields(self, field_list, realization_id):
        """
        Update hammurabi XML tree according to field list checklist.

        Parameters
        ----------

        realization_id : integer
            ID of realization in ensemble, [0,ensemble_size).
        """
        log.debug('@ hammurabi::update_fields')
        for field in field_list:
            # update physical parameters
            checklist = field.field_checklist
            paramlist = field.report_parameters(realization_id)
            for key, clue in checklist.items():
                assert (len(clue) == 2)
                self._ham.mod_par(clue[0], {clue[1]: str(paramlist[key])})
        
    def __call__(self, field_list):
        """
        Run hammurabi executable,
        then pack up outputs in IMAGINE convention.

        Parameters
        ----------

        field_list : list
            list of GeneralField objects

        Returns
        -------
        simulated results : Simulations object
        """
        log.debug('@ hammurabi::__call__')
        #t = Timer()
        #t.tick('simulator')
        self.register_fields(field_list)
        # execute hammurabi ensemble
        sims = Simulations()
        for i in range(self._ensemble_size):
            # update parameters
            self.update_fields(field_list, i)
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
