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
                                           self.prior,
                                           len(self._active_parameters),
                                           **self._sampling_controllers)

        self.sampler.run_nested(**kwargs)
        return self.sampler.results

    def _mpi_likelihood(self, cube):
        """
        mpi log-likelihood calculator
        Dynesty does not support execution with MPI
        where sampler on each node follows THE SAME journey in parameter space
        but not keep in communication
        so we calculate log-likelihood value of each node with joint force of all nodes
        in this way, ensemble size is multiplied by the number of working nodes

        Parameters
        ----------
        cube
            list of variable values

        Returns
        -------
        log-likelihood value
        """
        log.debug('@ dynesty_pipeline::_mpi_likelihood')
        # gather cubes from all nodes
        cube_local_size = cube.size
        cube_pool = np.empty(cube_local_size*mpisize, dtype=np.float64)
        comm.Allgather([cube, MPI.DOUBLE], [cube_pool, MPI.DOUBLE])
        # check if all nodes are at the same parameter-space position
        assert ((cube_pool == np.tile(cube_pool[:cube_local_size], mpisize)).all())
        return self._core_likelihood(cube)
