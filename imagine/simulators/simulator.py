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

    # converting time to int (ns level)
    def ct(self):
        return int(round(time.time()*1E+9))
    
    def seed_generator(self, seed):
        if seed > 0:
            return seed
        elif seed == 0:
            return self.ct()%int(1E+8) + threading.get_ident()%int(1E+8)
        else:
            raise ValueError('unsupported random seed value')
