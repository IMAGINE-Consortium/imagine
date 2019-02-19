from keepers import Loggable

'''
check pymultinest documentation for more instruction
'''
class Prior(Loggable, object):
    
    def __call__(self, cube, ndim, nparams):
        raise NotImplemented
