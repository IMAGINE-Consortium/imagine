"""
the mapper module is designed for implementing distribution mapping functions
"""

import numpy as np
import logging as log


def unity_mapper(x, a=0, b=1):
    """
    maps x from [0, 1] into the interval [a, b]
    
    parameters
    ----------
    
    x
        the variable to be mapped
        
    a
        the lower parameter value limit
        
    b
        the upper parameter value limit
    
    return
    ------
    
    the mapped parameter value
    """
    log.debug('@ carrier_mapper::unity_mapper')
    return np.float64(x) * (np.float64(b)-np.float64(a)) + np.float64(a)


def exp_mapper(x, a=0, b=1):
    """
    maps x from [0, 1] into the interval [exp(a), exp(b)]
    
    parameters
    ----------
    
    x
        the variable to be mapped
        
    a
        the lower parameter value limit
        
    b
        the upper parameter value limit
    
    return
    ------
    
    the mapped parameter value
    """
    log.debug('@ carrier_mapper::exp_mapper')
    return np.exp(unity_mapper(x, a, b))
