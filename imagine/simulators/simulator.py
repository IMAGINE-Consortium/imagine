"""
simulator base class
"""

from imagine.tools.icy_decorator import icy

@icy
class Simulator(object):

    def __init__(self):
        pass
    
    def __call__(self, field_list):
        raise NotImplementedError
