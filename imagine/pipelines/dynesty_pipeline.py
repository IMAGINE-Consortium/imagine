# %% IMPORTS
# Built-in imports
import os
import logging as log
import numpy as np
from multiprocessing import Pool
from os import path

# Package imports
import dynesty

# IMAGINE imports
from imagine.pipelines import Pipeline

# All declaration
__all__ = ['DynestyPipeline']


# %% CLASS DEFINITIONS
class DynestyPipeline(Pipeline):
    """
    Bayesian analysis pipeline with
    `dynesty <https://dynesty.readthedocs.io>`_

    This pipeline may use
    :py:class:`DynamicNestedSampler <dynesty.DynamicNestedSampler>` if the
    sampling parameter 'dynamic' is set to `True` or
    :py:class:`NestedSampler <dynesty.NestedSampler>`
    if 'dynamic` is False (default).

    See base class for initialization details.

    The sampler behaviour is controlled using the `sampling_controllers`
    property. A description of these can be found below.


    Sampling controllers
    --------------------
    dynamic : bool
        If `True`, use
        :py:class:`dynesty.DynamicNestedSampler` otherwise uses
        :py:class:`dynesty.NestedSampler`
    dlogz : float
        Iteration will stop, in the `dynamic==False` case,
        when the estimated contribution of the
        remaining prior volume to the total evidence falls below
        this threshold. Explicitly, the stopping criterion is
        `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
        evidence from all saved samples and `z_est` is the estimated
        contribution from the remaining volume. If `add_live` is `True`,
        the default is `1e-3 * (nlive - 1) + 0.01`. Otherwise, the
        default is `0.01`.
    dlogz_init : float
        The baseline run will stop, in the `dynamic==True` case,
        when the estimated contribution of the
        remaining prior volume to the total evidence falls below
        this threshold. Explicitly, the stopping criterion is
        `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
        evidence from all saved samples and `z_est` is the estimated
        contribution from the remaining volume. If `add_live` is `True`,
        the default is `1e-3 * (nlive - 1) + 0.01`. Otherwise, the
        default is `0.01`.
    dlogz_init : float
        If `dynamic` is `True`,  the *baseline run* will stop when the
        estimated contribution of the
        remaining prior volume to the total evidence falls below
        this threshold. Explicitly, the stopping criterion is
        `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
        evidence from all saved samples and `z_est` is the estimated
        contribution from the remaining volume. If `add_live` is `True`,
        the default is `1e-3 * (nlive - 1) + 0.01`. Otherwise, the
        default is `0.01`.
    nlive : int
        If `dynamic` is `False`, this sets the number of live points used.
        Default is 400.
    nlive_init : int
        If `dynamic` is `True`, this sets the number of live points used
        during the initial (“baseline”) nested sampling run. Default is 400.
    nlive_batch : int
        If `dynamic` is `True`, this sets the number of live points used
        when adding additional samples from a nested sampling run within
        each batch. Default is `400`.
    logl_max : float
        Iteration will stop when the sampled ln(likelihood) exceeds the
        threshold set by `logl_max`. Default is no bound (`np.inf`).
    logl_max_init : float
        The baseline run will stop, in the `dynamic==True` case, when the
        sampled ln(likelihood) exceeds this threshold.
        Default is no bound (`np.inf`).
    maxiter : int
        Maximum number of iterations.
        Iteration may stop earlier if the
        termination condition is reached. Default is (no limit).
    maxiter_init : int
        If `dynamic` is `True`, this sets the maximum number of iterations for
        the initial baseline nested sampling run. Iteration may stop earlier
        if the termination condition is reached. Default is sys.maxsize (no limit).
    maxiter_batch : int
        If `dynamic` is `True`, this sets the maximum number of iterations
        for the nested sampling run within each batch.
        Iteration may stop earlier if the termination condition is reached.
        Default is `sys.maxsize` (no limit).
    maxcall : int
        Maximum number of likelihood evaluations (without considering the
        initial points, i.e. maxcall_effective = maxcall + nlive).
        Iteration may stop earlier if termination condition is reached.
        Default is `sys.maxsize` (no limit).
    maxcall_init : int
        If `dynamic` is `True`, maximum number of likelihood evaluations in
        the baseline run.
    maxcall_batch : int
        If `dynamic` is `True`, maximum number of likelihood evaluations for
        the nested sampling run within each batch. Iteration may stop earlier
        if the termination condition is reached.
        Default is `sys.maxsize` (no limit).
    maxbatch : int
        If `dynamic` is `True`, maximum number of batches allowed.
        Default is `sys.maxsize` (no limit).
    use_stop : bool, optional
        Whether to evaluate our stopping function after each batch.
        Disabling this can improve performance if other stopping criteria
        such as :data:`maxcall` are already specified. Default is `True`.
    n_effective: int
        Minimum number of effective posterior samples. If the estimated
        "effective sample size" (ESS) exceeds this number,
        sampling will terminate. Default is no ESS (`np.inf`).
    n_effective_init: int
        Minimum number of effective posterior samples during the baseline run.
        If the estimated "effective sample size" (ESS) exceeds this number,
        sampling will terminate. Default is no ESS (`np.inf`).
    add_live : bool
        Whether or not to add the remaining set of live points to
        the list of samples at the end of each run. Default is `True`.
    print_progress : bool
        Whether or not to output a simple summary of the current run that
        updates with each iteration. Default is `True`.
    print_func : function
        A function that prints out the current state of the sampler.
        If not provided, the default :meth:`results.print_fn` is used.
    save_bounds : bool
        Whether or not to save past bounding distributions used to bound
          the live points internally. Default is *True*.
    bound : {`'none'`, `'single'`, `'multi'`, `'balls'`, `'cubes'`}
        Method used to approximately bound the prior using the current
        set of live points. Conditions the sampling methods used to
        propose new live points. Choices are no bound (`'none'`), a single
        bounding ellipsoid (`'single'`), multiple bounding ellipsoids
        (`'multi'`), balls centered on each live point (`'balls'`), and
        cubes centered on each live point (`'cubes'`). Default is `'multi'`.
    sample : {`'auto'`, `'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`}
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds. Unique methods available are:
        uniform sampling within the bounds(`'unif'`),
        random walks with fixed proposals (`'rwalk'`),
        random walks with variable ("staggering") proposals (`'rstagger'`),
        multivariate slice sampling along preferred orientations (`'slice'`),
        and "random" slice sampling along all orientations (`'rslice'`).
        `'auto'` selects the sampling method based on the dimensionality
        of the problem (from `ndim`).
        When `ndim < 10`, this defaults to `'unif'`.
        When `10 <= ndim <= 20`, this defaults to `'rwalk'`.
        When `ndim > 20`, this defaults to `'slice'`. `'rstagger'` and `'rslice'`
        are provided as alternatives for `'rwalk'` and `'slice'`, respectively.
        Default is `'auto'`. Note that Dynesty's 'hslice' option is not
        supported within IMAGINE.
    update_interval : int or float
        If an integer is passed, only update the proposal distribution every
        `update_interval`-th likelihood call. If a float is passed, update the
        proposal after every `round(update_interval * nlive)`-th likelihood
        call. Larger update intervals larger can be more efficient
        when the likelihood function is quick to evaluate. Default behavior
        is to target a roughly constant change in prior volume, with
        `1.5` for `'unif'`, `0.15 * walks` for `'rwalk'` and `'rstagger'`,
        `0.9 * ndim * slices` for `'slice'`, `2.0 * slices` for `'rslice'`,
        and `25.0 * slices` for `'hslice'`.
    enlarge : float
        Enlarge the volumes of the specified bounding object(s) by this
        fraction. The preferred method is to determine this organically
        using bootstrapping. If `bootstrap > 0`, this defaults to `1.0`.
        If `bootstrap = 0`, this instead defaults to `1.25`.
    bootstrap : int
        Compute this many bootstrapped realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes. Can lead
        to unstable bounding ellipsoids. Default is `0` (no bootstrap).
    vol_dec : float
        For the `'multi'` bounding option, the required fractional reduction
        in volume after splitting an ellipsoid in order to to accept the split.
        Default is `0.5`.
    vol_check : float
        For the `'multi'` bounding option, the factor used when checking if
        the volume of the original bounding ellipsoid is large enough to
        warrant `> 2` splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is `2.0`.
    walks : int
        For the `'rwalk'` sampling option, the minimum number of steps
        (minimum 2) before proposing a new live point. Default is `25`.
    facc : float
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.
    slices : int
        For the `'slice'` and `'rslice'` sampling
        options, the number of times to execute a "slice update"
        before proposing a new live point. Default is `5`.
        Note that `'slice'` cycles through **all dimensions**
        when executing a "slice update".


    Note
    ----
    Instances of this class are callable.
    Look at the :py:meth:`DynestyPipeline.call` for details.
    """

    def call(self, **kwargs):
        """
        Runs the IMAGINE pipeline using the Dynesty sampler

        Returns
        -------
        results : dict
                Dynesty sampling results
        """
        log.debug('@ dynesty_pipeline::__call__')

        # Resets internal state
        self.tidy_up()

        # Common parameters for NestedSampler and DynamicNestedSampler
        default_init_params = {'bound': 'multi',
                               'sample': 'auto',
                               'update_interval': None,
                               'enlarge' : None,
                               'bootstrap': 0,
                               'vol_dec': 0.5,
                               'facc': 0.5,
                               'slices': 5,
                               'walks': 25,
                               'vol_check': 2.0}

        # Keyword arguments can alter the sampling controllers
        self.sampling_controllers = kwargs  # Updates the dict

        dynamic = self.sampling_controllers.get('dynamic', True)

        resume = self.sampling_controllers.get('resume', False)
        assert (not resume), 'DynestyPipeline does not support resuming!'

        if dynamic:
            dynesty_sampler = dynesty.DynamicNestedSampler
            default_run_params = {'nlive_init': 400,
                                  'maxiter_init': None,
                                  'maxcall_init': None,
                                  'dlogz_init': 0.01,
                                  'logl_max_init': np.inf,
                                  'n_effective_init': np.inf,
                                  'nlive_batch': 400,
                                  'maxiter_batch': None,
                                  'maxcall_batch': None,
                                  'maxiter': None,
                                  'maxcall': None,
                                  'maxbatch': None,
                                  'n_effective': np.inf,
                                  'use_stop': True,
                                  'save_bounds': True,
                                  'print_progress': True,
                                  'print_func': None}
        else:
            dynesty_sampler = dynesty.NestedSampler
            default_init_params.update({'nlive': 400})
            default_run_params = {'maxiter': None,
                                  'maxcall': None,
                                  'dlogz': None,
                                  'logl_max': np.inf,
                                  'n_effective': None,
                                  'print_progress': True,
                                  'print_func': None,
                                  'save_bounds': True}

        # Prepares initialization and run parameters from
        # defaults and sampling controllers
        init_params = { k : self.sampling_controllers.get(k, default)
                       for  k, default in default_init_params.items()}
        run_params = { k : self.sampling_controllers.get(k, default)
                       for  k, default in default_run_params.items()}

        # Updates the sampling controllers to reflect what is being used
        self.sampling_controllers = init_params # Updates the dict
        self.sampling_controllers = run_params # Updates the dict

        rstate = np.random.RandomState(seed=self.master_seed)

        self.sampler = dynesty_sampler(self._likelihood_function,
                                       self.prior_transform,
                                       len(self._active_parameters),
                                       pool=None,  # parallel not working
                                       gradient=None,  # gradients not supported
                                       live_points=None,
                                       rstate=rstate,
                                       **init_params)

        self.sampler.run_nested(**run_params)

        self.results = self.sampler.results

        self._samples_array = self.results['samples']
        self._evidence = self.results['logz']
        self._evidence_err = self.results['logzerr']

        return self.results
