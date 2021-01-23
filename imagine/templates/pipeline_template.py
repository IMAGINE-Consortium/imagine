from imagine.pipelines import Pipeline
import numpy as np
import MY_SAMPLER  # Substitute this by your own code

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
        # Initializes a sampler object
        # Here we provide a list of common options
        self.sampler = MY_SAMPLER.Sampler(
            # Active parameter names can be obtained from
            param_names=self.active_parameters,
            # The likelihood function is available in
            loglike=self._likelihood_function,
            # Some samplers need a "prior transform function"
            prior_transform=self.prior_transform,
            # Other samplers need the prior PDF, which is
            prior_pdf=self.prior_pdf,
            # Sets the directory where the sampler writes the chains
            chains_dir=self.chains_directory,
            # Sets the seed used by the sampler
            seed=self.master_seed
            )

        # Most samplers have a `run` method, which should be executed
        self.sampling_controllers.update(kwargs)
        self.results = self.sampler.run(**self.sampling_controllers)

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


    def get_intermediate_results(self):
        # If the sampler saves intermediate results on disk or internally,
        # these should be read here, so that a progress report can be produced
        # every `pipeline.n_evals_report` likelihood evaluations.
        # For this to work, the following should be added to the
        # `intermediate_results` dictionary (currently commented out):

        ## Sets current rejected/dead points, as a numpy array of shape (n, npar)
        #self.intermediate_results['rejected_points'] = rejected
        ## Sets likelihood value of *rejected* points
        #self.intermediate_results['logLikelihood'] = likelihood_rejected
        ## Sets the prior volume/mass associated with each rejected point
        #self.intermediate_results['lnX'] = rejected_data[:, nPar+1]
        ## Sets current live nested sampling points (optional)
        #self.intermediate_results['live_points'] = live

        pass
