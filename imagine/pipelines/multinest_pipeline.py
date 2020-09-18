# %% IMPORTS
# Built-in imports
import logging as log
from os import path

# Package imports
import pymultinest
import numpy as np
import os

# IMAGINE imports
from imagine.pipelines import Pipeline

# All declaration
__all__ = ['MultinestPipeline']


# %% CLASS DEFINITIONS
class MultinestPipeline(Pipeline):
    """
    Bayesian analysis pipeline with
    `pyMultinest <https://github.com/JohannesBuchner/PyMultiNest>`_

    See base class for initialization details.

    The sampler behaviour is controlled using the `sampling_controllers`
    property. A description of these can be found below.

    Sampling controllers
    --------------------
    resume : bool
        If False the Pipeline the sampling starts from the beginning,
        overwriting any previous work in the `chains_directory`. Otherwise,
        tries to resume a previous run.
    n_live_points : int
        Number of live points to be used.
    evidence_tolerance : float
        A value of 0.5 should give good enough accuracy.
    max_iter : int
        Maximum number of iterations. `0` (default) is unlimited (i.e. only
        stops after convergence).
    log_zero : float
        Points with loglike < logZero will be ignored by MultiNest
    importance_nested_sampling : bool
        If `True` (default), Multinest will use Importance Nested
        Sampling (see `arXiv:1306.2144 <http://arxiv.org/abs/1306.2144>`_)
    sampling_efficiency : float
        Efficieny of the sampling. `0.8` (default) and `0.3` are recommended
        values for parameter estimation & evidence evaluation respectively.
    multimodal : bool
        If `True`, MultiNest will attempt to separate out the
        modes using a clustering algorithm.
    mode_tolerance : float
        MultiNest can find multiple modes and specify which samples belong to
        which mode. It might be desirable to have separate samples and mode
        statistics for modes with local log-evidence value greater than a
        particular value in which case `mode_tolerance` should be set to that
        value.
        If there isn't any particularly interesting `mode_tolerance` value,
        then it should be set to a very negative number (e.g. -1e90, default).
    null_log_evidence : float
        If `multimodal` is `True`, MultiNest can find multiple modes and also
        specify which samples belong to which mode. It might be
        desirable to have separate samples and mode statistics for modes
        with local log-evidence value greater than a
        particular value in which case nullZ should be set to that
        value. If there isn't any particulrly interesting
        nullZ value, then nullZ should be set to a very large negative
        number (e.g. -1.d90).
    n_clustering_params : int
        Mode separation is done through a clustering
        algorithm. Mode separation can be done on all the parameters
        (in which case nCdims should be set to ndims) & it
        can also be done on a subset of parameters (in which case
        nCdims < ndims) which might be advantageous as
        clustering is less accurate as the dimensionality increases.
        If nCdims < ndims then mode separation is done on
        the first nCdims parameters.
    max_modes : int
        Maximum number of modes (if `multimodal` is `True`).

    Note
    ----
    Instances of this class are callable.
    Look at the :py:meth:`MultinestPipeline.call` for details.
    """

    # Class attributes
    SUPPORTS_MPI = True

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

        default_solve_params = {'resume': False,
                                'n_live_points': 400,
                                'evidence_tolerance': 0.5,
                                'max_iter': 0,
                                'log_zero': -1e100,
                                'importance_nested_sampling': True,
                                'sampling_efficiency': 0.8,
                                'multimodal': True,
                                'mode_tolerance': -1e90,
                                'null_log_evidence': -1e90,
                                'n_clustering_params': None,
                                'max_modes': 100,
                                'n_iter_before_update': 100,
                                'outputfiles_basename': None,
                                'verbose': True}

        # Keyword arguments can alter the sampling controllers
        self.sampling_controllers = kwargs  # Updates the dict

        # Checks whether a base name for multinest output files was specified
        if 'outputfiles_basename' not in self.sampling_controllers:
            chains_prefix = path.join(self.chains_directory, 'multinest_')

            # If not, uses default location
            self.sampling_controllers['outputfiles_basename'] = chains_prefix

        # Prepares initialization and run parameters from
        # defaults and sampling controllers
        solve_params = {k: self.sampling_controllers.get(k, default)
                        for k, default in default_solve_params.items()}

        # Updates the sampling controllers to reflect what is being used
        self.sampling_controllers = solve_params  # Updates the dict

        if not self.sampling_controllers['resume']:
            self.clean_chains_directory()

        # Runs pyMultinest
        log.info('Calling pymultinest.solve')
        self.results = pymultinest.solve(LogLikelihood=self._likelihood_function,
                                         Prior=self.prior_transform,
                                         n_dims=len(self._active_parameters),
                                         wrapped_params=None,
                                         write_output=True,
                                         seed=self.master_seed,
                                         **solve_params)

        self._samples_array = self.results['samples']
        self._evidence = self.results['logZ']
        self._evidence_err = self.results['logZerr']

        return self.results

    def get_intermediate_results(self):

        nPar = len(self._active_parameters)


        if os.path.isfile(os.path.join(self.chains_directory, 'multinest_ev.dat')):
            live_data = np.genfromtxt(
                os.path.join(self.chains_directory, 'multinest_phys_live.points'))
            rejected_data = np.genfromtxt(
                os.path.join(self.chains_directory, 'multinest_ev.dat'))


            if len(rejected_data)>0:
                self.intermediate_results['rejected_points'] = rejected_data[:, :nPar]
                self.intermediate_results['live_points'] = live_data[:, :nPar]
                self.intermediate_results['logLikelihood'] = rejected_data[:, nPar]
                self.intermediate_results['lnX'] = rejected_data[:, nPar+1]
