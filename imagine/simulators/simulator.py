"""
simulator should be designed with

__init__ function taking Measurements keys
(for triggering output settings)
__call__ function taking list/tuple of field objects
(for triggering field settings)
returning a Simulations object

"""

import time
import threading

class Simulator(object):

    def __init__(self):
        pass
    
    def __call__(self, field_list):
        raise NotImplementedError
    
    def seed_generator(self, seed):
        # converting time to int (ns level)
        ct = lambda: int(round(time.time()*1E+9))
        if seed > 0:
            return seed
        elif seed == 0:
            return ct()%int(1E+8) + threading.get_ident()%int(1E+8)
        else:
            raise ValueError('unsupported random seed value')
