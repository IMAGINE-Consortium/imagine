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

    Note
    ----
    Instances of this class are callable. See `call` method.

    """
    def __init__(self, simulator, factory_list, likelihood, prior, ensemble_size=1):
        super(DynestyPipeline, self).__init__(simulator, factory_list, likelihood, prior, ensemble_size)

    def __call__(self, kwargs=dict()):
        return self.call(kwargs)

    def call(self, kwargs=dict()):
        """
        Parameters
        ----------
        kwargs : dict
            extra input argument controlling sampling process
            i.e., 'dlogz' for stopping criteria

        Returns
        ------
        Dynesty sampling results
        """
        log.debug('@ dynesty_pipeline::__call__')
        # init dynesty
        sampler = dynesty.NestedSampler(self._mpi_likelihood,
                                        self.prior,
                                        len(self._active_parameters),
                                        **self._sampling_controllers)
        sampler.run_nested(**kwargs)
        return sampler.results

    def _mpi_likelihood(self, cube):
        """
        mpi log-likelihood calculator
        PyMultinest supports execution with MPI
        where sampler on each node follows the same journey in parameter space
        but not keep in communication
        so we calculate log-likelihood value of each node with joint force of all nodes
        in this way, ensemble size is multiplied by the number of working nodes

        Parameters
        ----------
        cube
            list of variable values

        Returns
        -------
        Log-likelihood value
        """
        log.debug('@ multinest_pipeline::_mpi_likelihood')
        # gather cubes from all nodes
        cube_local_size = cube.size
        cube_pool = np.empty(cube_local_size*mpisize, dtype=np.float64)
        comm.Allgather([cube, MPI.DOUBLE], [cube_pool, MPI.DOUBLE])
        # check if all nodes are at the same parameter-space position
        assert ((cube_pool == np.tile(cube_pool[:cube_local_size], mpisize)).all())
        return self._core_likelihood(cube)

    def _core_likelihood(self, cube):
        """
        core log-likelihood calculator
        cube has been 'broadcasted' in the 2nd step in _mpi_likelihood
        now self._simulator will work on each node and provide multiple ensemble size
        """
        log.debug('@ multinest_pipeline::_core_likelihood')
        # security boundary check
        if np.any(cube > 1.) or np.any(cube < 0.):
            log.debug('cube %s requested. returned most negative possible number' % str(cube))
            return np.nan_to_num(-np.inf)
        # return active variables from pymultinest cube to factories
        # and then generate new field objects
        head_idx = int(0)
        tail_idx = int(0)
        field_list = tuple()
        # random seeds manipulation
        self._randomness()
        # the ordering in factory list and variable list is vital
        for factory in self._factory_list:
            variable_dict = dict()
            tail_idx = head_idx + len(factory.active_parameters)
            factory_cube = cube[head_idx:tail_idx]
            for i, av in enumerate(factory.active_parameters):
                variable_dict[av] = factory_cube[i]
            field_list += (factory.generate(variables=variable_dict,
                                            ensemble_size=self._ensemble_size,
                                            ensemble_seeds=self._ensemble_seeds),)
            log.debug('create '+factory.name+' field')
            head_idx = tail_idx
        assert(head_idx == len(self._active_parameters))
        observables = self._simulator(field_list)
        # apply mask
        observables.apply_mask(self.likelihood.mask_dict)
        # add up individual log-likelihood terms
        current_likelihood = self.likelihood(observables)
        # check likelihood value until negative (or no larger than given threshold)
        if self._check_threshold and current_likelihood > self._likelihood_threshold:
            raise ValueError('log-likelihood beyond threashould')
        return current_likelihood * self.likelihood_rescaler
