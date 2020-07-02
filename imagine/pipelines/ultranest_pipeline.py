# %% IMPORTS
# Built-in imports
import logging as log

# Package imports
import ultranest

# IMAGINE imports
from imagine.pipelines import Pipeline

# All declaration
__all__ = ['UltranestPipeline']


# %% CLASS DEFINITIONS
class UltranestPipeline(Pipeline):
    """
    Initialises Bayesian analysis pipeline with
    `UltraNest <https://johannesbuchner.github.io/UltraNest/>`_

    See base class for initialization details.

    Note
    ----
    Instances of this class are callable
    """

    # Class attributes
    SUPPORTS_MPI = True

    def call(self, **kwargs):
        """
        Runs the IMAGINE pipeline using the UltraNest sampler

        Returns
        -------
        results : dict
            UltraNest sampling results in a dictionary containing the keys:
            logZ (the log-evidence), logZerror (the error in log-evidence) and
            samples (equal weighted posterior)
        """
        log.debug('@ ultranest_pipeline::__call__')

        # Resets internal state
        self.tidy_up()

        # Runs UltraNest
        self.sampler = ultranest.ReactiveNestedSampler(
            param_names=list(self.active_parameters),
            loglike=self._likelihood_function,
            transform=self.prior_transform,
            #resume='subfolder',
            #run_num=None,
            #log_dir=None,
            #num_test_samples=2,
            #draw_multiple=True,
            #num_bootstraps=30,
            vectorized=False)

        kwargs_actual = {'viz_callback': False}
        kwargs_actual.update(self.sampling_controllers)
        kwargs_actual.update(kwargs)

        self.results = self.sampler.run(**kwargs_actual)

        self._samples_array = self.results['samples']
        self._evidence = self.results['logz']
        self._evidence_err = self.results['logzerr']

        return self.results
