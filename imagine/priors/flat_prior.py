"""
flat prior
"""

import logging as log
from imagine.priors.prior import Prior
from imagine.tools.icy_decorator import icy


@icy
class FlatPrior(Prior):

    def __init__(self):
        pass
    
    def __call__(self, cube):
        """
        return variable value as it is
        
        parameters
        ----------
        
        cube
            list of variable values in range [0,1]
            
        return
        ------
        list of variable values in range [0,1]
        """
        log.debug('@ flat_prior::__call__')
        return cube
