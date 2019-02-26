import logging as log

from imagine.priors.prior import Prior

"""
uniform prior with range [0,1]
check pymultinest documentation for more instruction
"""
class FlatPrior(Prior):

    def __init__(self):
        pass
    
    def __call__(self, cube):
        return cube
