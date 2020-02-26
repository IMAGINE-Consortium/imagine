from imagine.tools.icy_decorator import icy

@icy
class Prior(object):
    """
    Prior base class
    """
    def __init__(self):
        pass

    def __call__(self, cube):
        raise NotImplementedError
