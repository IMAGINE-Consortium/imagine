"""
built only for testing purpose

"""
#field model
    #mimicking emission intensity
    #field = square( gaussian_rand(mean=a,std=b)_x * sin(x) )

    #x in [0,2pi]
    #a and b are free parameters
import numpy as np

from imagine.simulators.simulator import Simulator
from imagine.fields.test_field.test_field import TestField
from imagine.observables.observable_dict import Measurements, Simulations
from imagine.tools.random_seed import seed_generator
from imagine.tools.icy_decorator import icy


@icy
class BiSimulator(Simulator):
    r"""
    Mock model mimicking Faraday rotation

    .. math::
           F = [\mathcal{G}(a, b) \sin(x)]^2

    where :math:`\mathcal{G}` is a Gaussian process with mean
    :math:`\mu=a` and standard deviation :math:`\sigma=b`, and
    :math:`x\in [0,2pi]`.

    Parameters
    ----------
    measurements
        Measurements object
        for testing, only key ('test',...,...,...) is valid

    Notes
    -----
    Instances of this class are callable
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

    def __call__(self, field_list):
        """
        Generates observables with parameter info from input field list

        Parameters
        ----------
        field_list
            list/tuple of field object

        Returns
        -------
        imagine.observables.observable_dict.Simulations
            Simulations object
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
        Applies field model and generate observable raw data

        Parameters
        ----------
        field_list
            list of field objects
        ensemble_size
            number of realizations in ensemble
        obs_size
            size of observable

        Returns
        -------
        numpy.ndarray
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
            raw_arr[i, :] = np.square(np.multiply(np.sin(coo_x),
                                                  np.random.normal(loc=par_a, scale=par_b, size=obs_size)))
        return raw_arr
