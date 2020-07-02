from imagine.pipelines import Pipeline
import numpy as np
import MY_SAMPLER

class PipelineTemplate(Pipeline):
    """
    Detailed description of sampler being adopted
    """

    # Class attributes
    # Does this sampler support MPI? Only necessary if True
    SUPPORTS_MPI = False

    def call(self, **kwargs):
        """
        Runs the IMAGINE pipeline

        Returns
        -------
        results : dict
            A dictionary containing the sampler results
            (usually in its native format)
        """
        # Resets internal state and adjusts random seed
        self.tidy_up()

        # Initializes a sampler object
        # Here we provide a list of common options
        self.sampler = MY_SAMPLER.sampler(
            # Active parameter names can be obtained from
            param_names=self.active_parameters,
            # The likelihood function is available in
            loglike=self._likelihood_function,
            # Some samplers need a "prior transform function"
            prior_transform=self.prior_transform,
            # Other samplers need the prior PDF, which is
            prior_pdf=self.prior_pdf)

        # Most samplers have a `run` method, which should be executed
        self.sampling_controllers.update(kwargs)
        self.results = self.sampler.run(self.sampling_controllers)

        # The samples should be converted to a numpy array and saved
        # to self._samples_array. This should have different samples
        # on different rows and each column corresponds to an active
        # parameter
        self._samples_array = self.results['samples']
        # The log of the computed evidence and its error estimate
        # should also be stored in the following way
        self._evidence = self.results['logz']
        self._evidence_err = self.results['logzerr']

        return self.results
