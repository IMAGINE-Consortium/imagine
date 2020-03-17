import logging as log
from imagine.priors.prior import GeneralPrior
from imagine.tools.icy_decorator import icy

@icy
class FlatPrior(GeneralPrior):
    """
    Flat prior
    """
    def __init__(self):
        pass

    def __call__(self, cube):
        """
        Return variable value as it is

        parameters
        ----------
        cube : list
            List of variable values in range [0,1]

        Returns
        -------
        List of variable values in range [0,1]
        """
        log.debug('@ flat_prior::__call__')
        return cube
