# %% IMPORTS
# Built-in imports
import logging as log
import shutil
import os
from os import path

# Package imports
import ultranest

# IMAGINE imports
from imagine.pipelines import Pipeline

# All declaration
__all__ = ['UltranestPipeline']


# %% CLASS DEFINITIONS
class UltranestPipeline(Pipeline):
    """
    Initializes Bayesian analysis pipeline with
    `UltraNest <https://johannesbuchner.github.io/UltraNest/>`_

    See base class for initialization details.

    The sampler behaviour is controlled using the `sampling_controllers`
    property. A description of these can be found below.

    Sampling controllers
        resume : bool
            If False the Pipeline the sampling starts from the beginning,
            erasing any previous work in the `chains_directory`. Otherwise,
            resume a previous run.
        dlogz : float
            Target evidence uncertainty. This is the std
            between bootstrapped logz integrators.
        dKL : float
            Target posterior uncertainty. This is the Kullback-Leibler
            divergence in nat between bootstrapped integrators.
        frac_remain : float
            Integrate until this fraction of the integral is left in the
            remainder.
            Set to a low number (1e-2 ... 1e-5) to make sure peaks are
            discovered.
            Set to a higher number (0.5) if you know the posterior is simple.
        Lepsilon : float
            Terminate when live point likelihoods are all the same,
            within Lepsilon tolerance. Increase this when your likelihood
            function is inaccurate, to avoid unnecessary search.
        min_ess : int
            Target number of effective posterior samples.
        max_iters : int
            maximum number of integration iterations.
        max_ncalls : int
            stop after this many likelihood evaluations.
        max_num_improvement_loops : int
            run() tries to assess iteratively where more samples are needed.
            This number limits the number of improvement loops.
        min_num_live_points : int
            minimum number of live points throughout the run
        cluster_num_live_points : int
            require at least this many live points per detected cluster
        num_test_samples : int
            test transform and likelihood with this number of
            random points for errors first. Useful to catch bugs.
        draw_multiple : bool
            draw more points if efficiency goes down.
            If set to False, few points are sampled at once.
        num_bootstraps : int
            number of logZ estimators and MLFriends region
            bootstrap rounds.
        update_interval_iter_fraction : float
            Update region after (update_interval_iter_fraction*nlive)
            iterations.

    Note
    ----
    Instances of this class are callable.
    Look at the :py:meth:`UltranestPipeline.call` for details.
    """

    # Class attributes
    SUPPORTS_MPI = True

    def call(self, **kwargs):
        """
        Runs the IMAGINE pipeline using the
        `UltraNest <https://johannesbuchner.github.io/UltraNest/>`_
        :py:class:`ReactiveNestedSampler <ultranest.integrator.ReactiveNestedSampler>`.

        Any keyword argument provided is used to update the
        `sampling_controllers`.

        Returns
        -------
        results : dict
            UltraNest sampling results in a dictionary containing the keys:
            logZ (the log-evidence), logZerror (the error in log-evidence) and
            samples (equal weighted posterior)

        Notes
        -----
        See base class for other attributes/properties and methods
        """
        log.debug('@ ultranest_pipeline::__call__')

        # Resets internal state
        self.tidy_up()

        default_init_params = {
          'resume': False,
          'num_test_samples': 2,
          'num_bootstraps': 30,
          'draw_multiple': True}

        default_run_params = {
          'dlogz': 0.5,
          'dKL': 0.5,
          'frac_remain': 0.01,
          'Lepsilon': 0.001,
          'min_ess': 500,
          'max_iters': None,
          'max_ncalls': None,
          'max_num_improvement_loops': -1,
          'min_num_live_points': 400,
          'cluster_num_live_points': 40,
          'update_interval_iter_fraction': 0.2}

        # Keyword arguments can alter the sampling controllers
        self.sampling_controllers = kwargs # Updates the dict

        # Prepares initialization and run parameters from
        # defaults and sampling controllers
        init_params = { k : self.sampling_controllers.get(k, default)
                       for  k, default in default_init_params.items()}
        run_params = { k : self.sampling_controllers.get(k, default)
                       for  k, default in default_run_params.items()}

        # Updates the sampling controllers to reflect what is being used
        self.sampling_controllers = init_params # Updates the dict
        self.sampling_controllers = run_params # Updates the dict

        # Ultranest files directory
        ultranest_dir = path.join(self.chains_directory, 'ultranest')
        # Creates directory, if needed
        os.makedirs(ultranest_dir, exist_ok=True)

        # Cleans up the chains directory if not resuming
        if not init_params['resume']:
            init_params['resume'] = 'overwrite'
            # Removing manually as UltraNest's 'overwrite' option does not
            # seem to be working correctly
            shutil.rmtree(ultranest_dir)
            # re-creates directory
            os.makedirs(ultranest_dir)

        # Creates directory, if needed
        os.makedirs(ultranest_dir, exist_ok=True)

        # Runs UltraNest
        self.sampler = ultranest.ReactiveNestedSampler(
            param_names=list(self.active_parameters),
            loglike=self._likelihood_function,
            transform=self.prior_transform,
            log_dir=ultranest_dir,
            vectorized=False,
            **init_params)

        self.results = self.sampler.run(viz_callback=False, **run_params)

        self._samples_array = self.results['samples']
        self._evidence = self.results['logz']
        self._evidence_err = self.results['logzerr']

        return self.results
