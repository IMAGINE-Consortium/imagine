from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

'''
field class register the full default parameter real time value
which is passed down from field factory, and prepare to hand in to simulators
the default checklist is a dict

members:
.simulator_checklist
    -- dict, with parameter name as entry, parameter xml path as content,
    defines the parameters to be checked out by simulator, should be fixed
    upon class design
    
'''
class GeneralField(object):

    def __init__(self, parameters=dict(), ensemble_size=1, random_seed=None):
        self.parameters = parameters
        self.ensemble_size = ensemble_size
        # if checklist has 'random_seed' entry
        if 'random_seed' in self.simulator_checklist:
            self.parameters.update({'random_seed':int(random_seed)})
    
    @property
    def simulator_checklist(self):
        return dict()

    @property
    def ensemble_size(self):
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        self._ensemble_size = int(ensemble_size)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = dict()
        # security check
        for k in parameters:
            assert (k in self.simulator_checklist)
        self._parameters.update(parameters)
