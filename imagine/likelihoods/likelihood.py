"""
Likelihood class defines likelihood posterior function
to be used in Bayesian analysis

members:
__init__
    -- initialisation requires
    Measurements object
    Covariances object
__call__
    -- running LOG-likelihood calculation requires
    ObservableDict object
"""

from imagine.tools.icy_decorator import icy

@icy
class Likelihood(object):

    def __init__(self):
        pass
    
    def __call__(self, observable_dict):
        raise NotImplementedError
