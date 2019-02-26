"""
Likelihood class defines likelihood posterior function
to be used in Bayesian analysis

members:
._to_global_data
    -- works trivially on list/tuple
__init__
    -- initialisation requires
    Measurements object
    Covariances object
__call__
    -- running LOG-likelihood calculation requires
    ObservableDict object

"""

import numpy as np
import logging as log

from nifty5 import Field

from imagine.observables.observable import Observable

class Likelihood(object):

    def __init__(self):
        pass
    
    def __call__(self, observable_dict):
        raise NotImplementedError
