"""
Prior base class is designed as a shell
check pymultinest documentation for more instruction
"""
class Prior(object):
    
    def __call__(self, cube):
        raise NotImplementedError
