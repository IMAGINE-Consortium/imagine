from keepers import Loggable

'''
simulator should be designed with __call__ function
taking dict of field objects in the form
{field_name: field object}

fild_name is from field_factory.name
field object should be produced by field_factory.generate function
which carries information of parameter values, which to be hand in
to simulators
'''
class Simulator(Loggable, object):
    
    def __call__(field_dict):
        raise NotImplementedError
