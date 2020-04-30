import logging as log
import dynesty
import numpy as np
from mpi4py import MPI
from imagine.pipelines.pipeline import Pipeline
from imagine.tools.icy_decorator import icy


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

@icy
class DynestyPipeline(Pipeline):
    """
    Initialises Bayesian analysis pipeline with Dynesty

    See base class for initialization details.
    """
    @property
    def sampler_supports_mpi(self):
        return False

    def __call__(self, dynamic=True, **kwargs):
        """
        Parameters
        ----------
        dynamic : bool
            Uses *dynamic* nested sampling, i.e. adjust the number of
            live points on the fly to achieve optimal description of the
            posterior. If False, regular (static) nested sampling is used
            (with the number of live points specified by `nlive`).
        **kwargs : Dynesty parameters
            Extra input argument controlling sampling process
            i.e., 'nlive' for stopping criteria

        Returns
        -------
        Dynesty sampling results
        """
        log.debug('@ dynesty_pipeline::__call__')

        if self.sampler is None:
            # init dynesty
            if dynamic:
                dynesty_sampler = dynesty.DynamicNestedSampler
            else:
                dynesty_sampler = dynesty.NestedSampler

            self.sampler = dynesty_sampler(self._mpi_likelihood,
                                           self.prior_transform,
                                           len(self._active_parameters),
                                           **self._sampling_controllers)

        self.sampler.run_nested(**kwargs)

        self.sampler.results = self.sampler.results

        self._samples_array = results['samples']
        self._evidence = results['logz']
        self._evidence_err = results['logzerr']

        return results
