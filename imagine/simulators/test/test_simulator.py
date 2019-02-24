'''
test simulator is built only for testing purpose
'''

import numpy as np
import logging as log
import time
import threading

from imagine.simulators.simulator import Simulator
from imagine.fields.test_field.test_field import TestField
from imagine.observables.observable_dict import Measurements, Simulations

class TestSimulator(Simulator):

    '''
    measurements
        -- Measruments object
        for testing, only key ('test',...,...,...) is valid
    '''
    def __init__(self, measurements):
        self.output_checklist = measurements

    @property
    def output_checklist(self):
        return self._output_checklist

    @output_checklist.setter
    def output_checklist(self, measurements):
        assert isinstance(measurements, Measurements)
        self._output_checklist = tuple(measurements.keys())

    '''
    filed_list
        -- a list/tuple of field object
    '''
    def __call__(self, field_list):
        assert (len(self._output_checklist) == 1)
        assert (self._output_checklist[0][0] == 'test')
        obsdim = int(self._output_checklist[0][2])
        # check input
        assert isinstance(field_list, (list,tuple))
        assert (len(field_list) == 1)
        assert isinstance(field_list[0], TestField)
        ensize = field_list[0].ensemble_size
        pars = field_list[0].parameters
        # double check parameter keys
        assert (pars.keys() == field_list[0].field_checklist.keys())
        # assemble Simulations object
        output = Simulations()
        # core function for producing observables
        obs_arr = self.obs_generator(pars,ensize, obsdim)
        # not using healpix structure
        output.append(self._output_checklist[0],obs_arr,True)
        return output

    def obs_generator(self, parameters, ensemble_size, length):
        # extract parameters
        par_a = parameters['a']
        par_b = parameters['b']
        par_s = parameters['random_seed']
        # converting time to int (ns level)
        ct = lambda: int(round(time.time()*1E+9))
        # get thread-time dependent random number
        # if given seed is zero
        if (par_s>0):
            np.random.seed(par_s)
        elif (par_s == 0):
            # numpy seed int has 32 bit
            np.random.seed(ct()%int(1E+8) + threading.get_ident()%int(1E+8))
        else:
            raise ValueError('unsupported random seed value')
        # coordinates
        out_arr = np.zeros((ensemble_size,length))
        coo_x = np.linspace(0,1,length)
        for i in range(ensemble_size):
            out_arr[i] = par_a*coo_x**2 + np.random.normal(scale=par_b,size=length)
        return out_arr
