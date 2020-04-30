import numpy as np
import logging as log
import os
import pymultinest
from mpi4py import MPI
from imagine.pipelines.pipeline import Pipeline
from imagine.tools.icy_decorator import icy


comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

@icy
class MultinestPipeline(Pipeline):
    """
    Initialises Bayesian analysis pipeline with pyMultinest

    See base class for initialization details.

    Note
    ----
    Instances of this class are callable

    """
    @property
    def sampler_supports_mpi(self):
        return True

    def __call__(self, **kwargs):
        """
        Returns
        -------
        results : dict
            pyMultinest sampling results in a dictionary containing the keys:
            logZ (the log-evidence), logZerror (the error in log-evidence) and
            samples (equal weighted posterior)
        """
        log.debug('@ multinest_pipeline::__call__')

        # Checks whether a base name for multinest output files was specified
        if 'outputfiles_basename' not in self._sampling_controllers:
            # If not, uses default location
            self._sampling_controllers['outputfiles_basename'] = 'chains/imagine_'
            os.makedirs('chains', exist_ok=True)

        # Makes sure that the chains directory exists
        basedir = os.path.split(self._sampling_controllers['outputfiles_basename'])[0]
        assert os.path.isdir(basedir)

        # Runs pyMultinest
        results = pymultinest.solve(LogLikelihood=self._mpi_likelihood,
                                    Prior=self.prior_transform,
                                    n_dims=len(self._active_parameters),
                                    **self._sampling_controllers)

        self._samples_array = results['samples']
        self._evidence = results['logz']
        self._evidence_err = results['logzerr']

        return results
