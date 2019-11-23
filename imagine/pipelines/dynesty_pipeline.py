import numpy as np
import logging as log
import dynesty
from mpi4py import MPI
from imagine.pipelines.pipeline import Pipeline
from imagine.tools.icy_decorator import icy

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


@icy
class DynestyPipeline(Pipeline):

    def __init__(self, simulator, factory_list, likelihood, prior, ensemble_size=1):
        """
        """
        super(DynestyPipeline, self).__init__(simulator, factory_list, likelihood, prior, ensemble_size)

    def __call__(self, kwargs=dict()):
        """

        :param kwargs: extra input argument controlling sampling process
        i.e., 'dlogz' for stopping criteria
        :return: Dynesty sampling results
        """
        if mpisize > 1:
            raise ValueError('MPI unsupported in Dynesty')
        # init dynesty
        sampler = dynesty.NestedSampler(self._core_likelihood,
                                        self.prior,
                                        len(self._active_parameters),
                                        **self._sampling_controllers)
        sampler.run_nested(**kwargs)
        return sampler.results
