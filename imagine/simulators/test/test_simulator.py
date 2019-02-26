"""
test simulator is built only for testing purpose

field model:
    y = a*sin(x) + (gaussian_rand(mean=0,std_div=b))**2
    x in [0,2pi]
    a and b are free parameters
    so test field model contains a 'sin' smooth/regular field
    and a variance-like influence from gaussian random field
"""

import numpy as np
import logging as log

from imagine.simulators.simulator import Simulator
from imagine.fields.test_field.test_field import TestField
from imagine.observables.observable_dict import Measurements, Simulations

class TestSimulator(Simulator):

    """
    measurements
        -- Measurements object
        for testing, only key ('test',...,...,...) is valid
    """
    def __init__(self, measurements):
        self.output_checklist = measurements

    @property
    def output_checklist(self):
        return self._output_checklist

    @output_checklist.setter
    def output_checklist(self, measurements):
        assert isinstance(measurements, Measurements)
        self._output_checklist = tuple(measurements.keys())

    """
    filed_list
        -- a list/tuple of field object
    """
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
        # get thread-time dependent random number
        np.random.seed(self.seed_generator(par_s))
        # coordinates
        out_arr = np.zeros((ensemble_size,length))
        coo_x = np.linspace(0.,2.*np.pi,length)
        for i in range(ensemble_size):
            out_arr[i,:] = par_a*np.sin(coo_x) + (np.random.normal(0.,par_b,length))**2
        return out_arr
