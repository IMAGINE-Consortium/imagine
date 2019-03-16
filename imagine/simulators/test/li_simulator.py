"""
built only for testing purpose

field model:
    mimicking Faraday rotation
    field = gaussian_rand(mean=a,std=b)_x * cos(x)

    x in [0,2pi]
    a and b are free parameters
"""

import numpy as np

from imagine.simulators.simulator import Simulator
from imagine.fields.test_field.test_field import TestField
from imagine.observables.observable_dict import Measurements, Simulations
from imagine.tools.random_seed import seed_generator
from imagine.tools.icy_decorator import icy


@icy
class LiSimulator(Simulator):

    def __init__(self, measurements):
        """

        :param measurements: Measurements object
        for testing, only key ('test',...,...,...) is valid
        """
        self.output_checklist = measurements

    @property
    def output_checklist(self):
        return self._output_checklist

    @output_checklist.setter
    def output_checklist(self, measurements):
        assert isinstance(measurements, Measurements)
        self._output_checklist = tuple(measurements.keys())

    def __call__(self, field_list):
        """
        generate observables with parameter info from input field list
        :param field_list: list/tuple of field object
        :return: Simulations object
        """
        assert (len(self._output_checklist) == 1)
        assert (self._output_checklist[0][0] == 'test')
        obsdim = int(self._output_checklist[0][2])
        # check input
        assert isinstance(field_list, (list, tuple))
        assert (len(field_list) == 1)
        assert isinstance(field_list[0], TestField)
        ensize = field_list[0].ensemble_size
        # assemble Simulations object
        output = Simulations()
        # core function for producing observables
        obs_arr = self.obs_generator(field_list, ensize, obsdim)
        # not using healpix structure
        output.append(self._output_checklist[0], obs_arr, True)
        return output

    def obs_generator(self, field_list, ensemble_size, obs_size):
        """
        apply field model and generate observable raw data
        :param parameters: dict of parameters
        :param ensemble_size: number of realizations in ensemble
        :param obs_size: size of observable
        :return: numpy ndarray
        """
        # coordinates
        raw_arr = np.zeros((ensemble_size, obs_size))
        coo_x = np.linspace(0., 2.*np.pi, obs_size)
        for i in range(ensemble_size):
            pars = field_list[0].report_parameters(i)
            # double check parameter keys
            assert (pars.keys() == field_list[0].field_checklist.keys())
            # extract parameters
            par_a = pars['a']
            par_b = pars['b']
            par_s = pars['random_seed']
            # get thread-time dependent random number
            np.random.seed(seed_generator(par_s))
            raw_arr[i, :] = np.multiply(np.cos(coo_x),
                                        np.random.normal(loc=par_a, scale=par_b, size=obs_size))
        return raw_arr
