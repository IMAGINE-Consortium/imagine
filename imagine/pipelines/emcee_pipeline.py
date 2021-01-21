from imagine.pipelines import Pipeline
import numpy as np
import emcee

class EmceePipeline(Pipeline):
    """
    Detailed description of sampler being adopted
    """


    def log_probability(self, pvalues):
        # Checks whether the range is sane
        for pname, v in zip(self.active_parameters, pvalues):
            pmin, pmax = self.priors[pname].range.value
            if not (pmin < v < pmax):
                return -np.inf

        # Computes the (log) prior
        lp = np.log(self.prior_pdf(pvalues))
        if not np.isfinite(lp):
            return -np.inf

        # Multiplies by the likelihood
        log_prob = lp + self._likelihood_function(pvalues)

        return log_prob

    def call(self, max_iterations=10000, nwalkers=32, n_corr_times=100,
             **kwargs):
        """
        Runs the IMAGINE pipeline

        Sampling controllers
        --------------------
        nwalkers : int
            Number of walkers
        max_iterations : int
            Maximum number of iterations
        resume : bool
            If True, a previous state is used to resume the run.

        Returns
        -------
        results : dict
            A dictionary containing the sampler results
            (usually in its native format)
        """
        n_pars = len(self.active_parameters)
        # Prepares the initial positions of the priors
        pos = [self.prior_transform(np.random.sample(n_pars))
               for _ in range(nwalkers)]

        ndim = len(self._active_parameters)

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)

        old_tau = np.inf
        index = 0
        self._autocorr = np.empty(max_iterations)
        for sample in self.sampler.sample(pos, iterations=max_iterations,
                                     progress=True, store=True):
            if self.sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = self.sampler.get_autocorr_time(tol=0)
            self._autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * n_corr_times < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

        tau = self.sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))

        # Extract the flattened samples
        self._samples_array = sampler.get_chain(discard=burnin,
                                                thin=thin, flat=True)

        # The log of the computed evidence and its error estimate
        # should also be stored in the following way
        self._evidence = np.nan
        self._evidence_err = np.nan

        return self.results


    def get_intermediate_results(self):

        chain = self.sampler.get_chain(flat=True)

        # Reconstructs likelihood
        posterior = log_prob = self.sampler.get_log_prob(flat=True)
        prior = np.array([self.prior_pdf(point) for point in chain])
        logLikelihood = self.sampler.get_log_prob(flat=True) - np.log(prior)

        # Sets current rejected/dead points, as a numpy array of shape (n, npar)
        self.intermediate_results['rejected_points'] = chain
        # Sets likelihood value of *rejected* points
        self.intermediate_results['logLikelihood'] = logLikelihood

