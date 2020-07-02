# %% IMPORTS
# Built-in imports
import logging as log

# Package imports
import dynesty

# IMAGINE imports
from imagine.pipelines import Pipeline

# All declaration
__all__ = ['DynestyPipeline']


# %% CLASS DEFINITIONS
class DynestyPipeline(Pipeline):
    """
    Initialises Bayesian analysis pipeline with Dynesty

    See base class for initialization details.
    """

    def call(self, dynamic=True, **kwargs):
        """
        Runs the IMAGINE pipeline using the Dynesty sampler

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

        # Resets internal state
        self.tidy_up()

        if dynamic:
            dynesty_sampler = dynesty.DynamicNestedSampler
        else:
            dynesty_sampler = dynesty.NestedSampler

        self.sampler = dynesty_sampler(self._likelihood_function,
                                       self.prior_transform,
                                       len(self._active_parameters),
                                       **self._sampling_controllers)

        self.sampler.run_nested(**kwargs)

        self.results = self.sampler.results

        self._samples_array = self.results['samples']
        self._evidence = self.results['logz']
        self._evidence_err = self.results['logzerr']

        return self.results
