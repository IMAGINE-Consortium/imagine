# %% IMPORTS
# Built-in imports
import logging as log
import os
from os import path
import tempfile

# Package imports
from mpi4py import MPI
import pymultinest

# IMAGINE imports
from imagine.pipelines import Pipeline
import imagine as img

# GLOBALS
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

# All declaration
__all__ = ['MultinestPipeline']


# %% CLASS DEFINITIONS
class MultinestPipeline(Pipeline):
    """
    Initialises Bayesian analysis pipeline with pyMultinest

    See base class for initialization details.

    Note
    ----
    Instances of this class are callable
    """

    # Class attributes
    SUPPORTS_MPI = True

    def __init__(self, simulator, factory_list, likelihood, ensemble_size=1,
                 chains_directory=None):
        super().__init__(
            simulator=simulator,
            factory_list=factory_list,
            likelihood=likelihood,
            ensemble_size=ensemble_size)

        if chains_directory is not None:
            self._chains_dir_path = chains_directory

        else:
            if mpirank==0:
                # Creates a safe temporary directory in the current working directory
                self._chains_dir_obj = tempfile.TemporaryDirectory(prefix='imagine_chains_',
                                                                   dir=os.getcwd())
                # Note: this dir is automatically deleted together with the simulator object
                dir_path = self._chains_dir_obj.name
            else:
                dir_path = None

            self._chains_dir_path = comm.bcast(dir_path, root=0)


        self._chains_prefix = path.join(self._chains_dir_path,'')

    def call(self, **kwargs):
        """
        Runs the IMAGINE pipeline using the MultiNest sampler

        Returns
        -------
        results : dict
            pyMultinest sampling results in a dictionary containing the keys:
            logZ (the log-evidence), logZerror (the error in log-evidence) and
            samples (equal weighted posterior)
        """
        log.debug('@ multinest_pipeline::__call__')

        # Resets internal state
        self.tidy_up()

        # Checks whether a base name for multinest output files was specified
        if 'outputfiles_basename' not in self._sampling_controllers:
            # If not, uses default location
            self._sampling_controllers['outputfiles_basename'] = self._chains_prefix

        kwargs_actual = {}
        kwargs_actual.update(self.sampling_controllers)
        kwargs_actual.update(kwargs)

        # Runs pyMultinest
        self.results = pymultinest.solve(LogLikelihood=self._likelihood_function,
                                         Prior=self.prior_transform,
                                         n_dims=len(self._active_parameters),
                                         **kwargs_actual)

        self._samples_array = self.results['samples']
        self._evidence = self.results['logZ']
        self._evidence_err = self.results['logZerr']

        return self.results
