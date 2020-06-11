import logging as log
import os, os.path
import pymultinest
from imagine.pipelines.pipeline import Pipeline
from imagine.tools.icy_decorator import icy
import imagine as img
import tempfile

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

    def __init__(self, simulator, factory_list, likelihood, ensemble_size=1,
                 chains_directory=None):
        super().__init__(simulator, factory_list, likelihood, ensemble_size)

        if chains_directory is not None:
            self._chains_dir_path = chains_directory

        else:
            # Creates a safe temporary directory in the current working directory
            self._chains_dir_obj = tempfile.TemporaryDirectory(prefix='imagine_chains_',
                                                               dir=os.getcwd())
            # Note: this dir is automatically deleted together with the simulator object
            self._chains_dir_path = self._chains_dir_obj.name

        self._chains_prefix = os.path.join(self._chains_dir_path,'')

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
