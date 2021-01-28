# %% IMPORTS
# Built-in imports
import os
import logging


# Package imports
import schwimmbad
import numpy as np
import emcee
from mpi4py import MPI

# IMAGINE imports
from imagine.pipelines import Pipeline

# GLOBALS
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

# All declaration
__all__ = ['EmceePipeline']

class EmceePipeline(Pipeline):
    """
    Analysis pipeline with the MCMC sampler `emcee <https://github.com/dfm/emcee/>`_

    See base class for initialization details.

    The chains are considered converged once the total number of iterations
    becomes smaller than `convergence_factor` times the autocorrelation time.

    The sampler behaviour is controlled using the `sampling_controllers`
    property. A description of these can be found below.

    Sampling controllers
    --------------------
    resume : bool
        If False the Pipeline the sampling starts from the beginning,
        overwriting any previous work in the `chains_directory`. Otherwise,
        tries to resume a previous run.
    nwalkers : int
        Number of walkers
    max_nsteps : int
        Maximum number of iterations
    nsteps_check : int
        The sampler will check for convergence every `nsteps_check`
    convergence_factor : float
        Factor used to compute the convergence
    burnin_factor : int
        Number of autocorrelation times to be discarded from main results
    thin_factor : float
        Factor used to choose how the chain will be "thinned" after running

    """

    # Class attributes
    SUPPORTS_MPI = True

    def call(self, **kwargs):
        """

        Returns
        -------
        results : dict
            A dictionary containing the sampler results
            (usually in its native format)
        """
        logging.debug('@ emcee_pipeline::__call__')

        default_params = dict(max_nsteps=100000,
                              nwalkers=32,
                              burnin_factor=2,
                              thin_factor=0.5,
                              nsteps_check=100,
                              convergence_factor=100,
                              resume=True)

        # Keyword arguments can alter the sampling controllers
        self.sampling_controllers = kwargs  # Updates the dict

        # Prepares initialization and run parameters from
        # defaults and sampling controllers
        params = {k: self.sampling_controllers.get(k, default)
                        for k, default in default_params.items()}

        # Updates the sampling controllers to reflect what is being used
        self.sampling_controllers = params  # Updates the dict

        ndim = len(self._active_parameters)
        if mpisize==1:
            pool = None
            master = True
        else:
            # Uses an MPI pool with more than 1 process is present
            pool = schwimmbad.MPIPool(use_dill=True)
            master = pool.is_master()

        if not master:
            # "Worker" processes behaviour
            pool.wait()
            self.sampler = None
            self._samples_array = None
        else:
            filename = os.path.join(self.chains_directory,
                                    'chains.hdf5'.format(mpirank))

            backend = emcee.backends.HDFBackend(filename)

            # Prepares the initial positions of the priors
            pos = [self.prior_transform(np.random.sample(ndim))
                  for _ in range(params['nwalkers'])]
            # Only uses this if not resuming
            if os.path.isfile(filename):
                if self.sampling_controllers['resume']:
                    pos = None
                else:
                    backend.reset()

            # Sets up the sampler
            self.sampler = emcee.EnsembleSampler(params['nwalkers'], ndim,
                                                self.log_probability_unnormalized,
                                                pool=pool, backend=backend)

            old_tau = np.inf
            nsteps = 0

            # Iterates trying to reach convergence
            while nsteps < params['max_nsteps']:
                self.sampler.run_mcmc(pos, progress=True, store=True,
                                      nsteps=self.sampling_controllers['nsteps_check'])

                pos = None  # Resumes from where it stopped in next steps

                # Checks convergence (following emcee's authors prescription)
                self.tau = self.sampler.get_autocorr_time(tol=0)
                self.converged = np.all(
                  self.tau * params['convergence_factor'] < self.sampler.iteration)
                self.converged &= np.all(np.abs(old_tau - self.tau) / self.tau < 0.01)
                if self.converged:
                    break
                old_tau = self.tau
                nsteps = self.sampler.iteration

            burnin = int(params['burnin_factor'] * np.max(self.tau))
            thin = int(params['thin_factor'] * np.min(self.tau))

            self._samples_array = self.sampler.get_chain(discard=burnin,
                                                         thin=thin,
                                                         flat=True)
        if mpisize > 1:
            pool.close()
            self._samples_array = comm.bcast(self._samples_array, root=0)

        # This involves no computation of the evidence
        self._evidence = np.nan
        self._evidence_err = np.nan

        # Removes the internal reference to the sampler
        # (to avoid problems while saving)
        sampler, self.sampler = self.sampler, None
        # Returns the sampler object (to allow further checks by the user)
        return sampler


    def get_intermediate_results(self):
        if self.sampler is not None:
            chain = self.sampler.get_chain(flat=True)

            # Reconstructs likelihood
            posterior = log_prob = self.sampler.get_log_prob(flat=True)
            prior = np.array([self.prior_pdf(point) for point in chain])
            logLikelihood = self.sampler.get_log_prob(flat=True) - np.log(prior)

            # Sets current rejected/dead points, as a numpy array of shape (n, npar)
            self.intermediate_results['rejected_points'] = chain
            # Sets likelihood value of *rejected* points
            self.intermediate_results['logLikelihood'] = logLikelihood
