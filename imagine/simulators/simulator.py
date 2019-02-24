'''
simulator should be designed with

__init__ function taking Measurements keys
(for trigering output settings)
__call__ function taking list/tuple of field objects
(for trigering field settings)
returning a Simulations object

'''

class Simulator(object):

    def __init__(self, measurements):
        raise NotImplementedError
    
    def __call__(self, field_list):
        raise NotImplementedError
