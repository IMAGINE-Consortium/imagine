"""
provide time-thread dependent seed value
"""

import time
import threading


def seed_generator(_seed):
    """
    given 0 will trigger time-thread dependent method
    :param _seed: pre-fixed seed value
    :return: modified seed value
    """
    if _seed > 0:
        return _seed
    elif _seed == 0:
        return round(time.time()*1E+9) % int(1E+8) + threading.get_ident() % int(1E+8)
    else:
        raise ValueError('unsupported random seed value')
