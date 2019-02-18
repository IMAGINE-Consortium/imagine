class _MPI(object):
    
    def __init__(self):
        self.COMM_WORLD = _COMM_WORLD()

class _COMM_WORLD(object):
    
    def Get_size(self):
        return 1

MPI = _MPI()
