"""
this module provides time-thread dependent seed value.
For the testing suits, please turn to "imagine/tests/tools_tests.py".
"""

import numpy as np
import time
import threading
import logging as log


def seed_generator(trigger):
    """
    set trigger as 0 will generate time-thread dependent method
    otherwise return the trigger as seed
    
    Parameters
    ----------
    
    trigger : int
        non-negative value
        pre-fixed seed value
        
    Returns
    -------
    a random seed value
    """
    log.debug('@ random_seed::seed_generator')
    if trigger > 0:
        return int(trigger)
    elif trigger == 0:
        return round(time.time()*1E+9) % int(1E+8) + threading.get_ident() % int(1E+8)
    else:
        raise ValueError('unsupported random seed value')

def ensemble_seed_generator(size):
    """
    generate fixed random seed values for each realization in ensemble
    
    Parameters
    ----------
    
    size : int
        number of realizations in ensemble
        
    return
    ------
    numpy.ndarray
    a list of random seeds
    """
    log.debug('@ random_seed::ensemble_seed_generator')
    # the uint32 is defined by the random generator's capasity
    return np.random.randint(low=1, high=np.uint32(-1)//3, size=np.uint64(size))
