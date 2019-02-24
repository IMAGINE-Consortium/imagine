'''
simulator should be designed with

__init__ function taking Measurements keys
(for trigering output settings)
__call__ function taking list/tuple of field objects
(for trigering field settings)

'''

class Simulator(object):

    def __init__(self, measurement_keys):
        raise NotImplementedError
    
    def __call__(self, field_list):
        raise NotImplementedError
