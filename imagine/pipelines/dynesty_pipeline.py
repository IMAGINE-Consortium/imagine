import logging as log
import dynesty
from imagine.pipelines.pipeline import Pipeline
from imagine.tools.icy_decorator import icy


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

        return results
