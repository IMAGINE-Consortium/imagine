from imagine.priors.prior import Prior

'''
uniform prior with range [0,1]
check pymultinest documentation for more instruction
'''
class FlatPrior(Prior):
    
    def __call__(self, cube, ndim, nparams):
        return cube
