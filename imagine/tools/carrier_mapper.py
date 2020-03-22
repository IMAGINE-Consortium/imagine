"""
The mapper module is designed for implementing distribution mapping functions.
"""

import numpy as np
import logging as log


def unity_mapper(x, a=0., b=1.):
    """
    Maps x from [0, 1] into the interval [a, b].

    Parameters
    ----------
    x : float
        The variable to be mapped.
    a : float
        The lower parameter value limit.
    b : float
        The upper parameter value limit.

    Returns
    -------
    numpy.float64
        The mapped parameter value.
    """
    log.debug('@ carrier_mapper::unity_mapper')
    return np.float64(x) * (np.float64(b)-np.float64(a)) + np.float64(a)


def exp_mapper(x, a=0, b=1):
    """
    Maps x from [0, 1] into the interval [exp(a), exp(b)].

    Parameters
    ----------
    x : float
        The variable to be mapped.
    a : float
        The lower parameter value limit.
    b : float
        The upper parameter value limit.

    Returns
    -------
    numpy.float64
        The mapped parameter value.
    """
    log.debug('@ carrier_mapper::exp_mapper')
    return np.exp(unity_mapper(x, a, b))
