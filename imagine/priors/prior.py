"""
Prior base class
"""

from imagine.tools.icy_decorator import icy


@icy
class Prior(object):

    def __init__(self):
        pass
    
    def __call__(self, cube):
        raise NotImplementedError
