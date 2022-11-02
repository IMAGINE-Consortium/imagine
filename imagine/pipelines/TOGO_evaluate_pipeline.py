import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import os

from e13tools import q2tex
from mpi4py import MPI

from imagine.tools import visualization, misc

# GLOBALS
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()


def samples_summary(samples):
    r"""
    A dictionary containing a summary of posterior statistics for each of
    the active parameters.  These are: 'median', 'errlo'
    (15.87th percentile), 'errup' (84.13th percentile), 'mean' and 'stdev'.
    """

    posterior_summary = {}

    for name, column in zip(samples.columns, samples.itercols()):
        lo, median, up = np.percentile(column, [15.865, 50, 84.135])
        errlo = abs(median-lo)
        errup = abs(up-median)
        posterior_summary[name] = {
            'median': median,
            'errlo': errlo,
            'errup': errup,
            'mean': np.mean(column),
            'stdev': np.std(column)}

    return posterior_summary


def corner_plot(pipeline, **kwargs):
    """
    Calls :py:func:`imagine.tools.visualization.corner_plot` to make
    a corner plot of samples produced by this Pipeline
    """
    return visualization.corner_plot(pipeline=pipeline, **kwargs)


def progress_report(pipeline):
    """
    Reports the progress of the inference
    """
    # Try to call get_intermediate_results
    try:
        pipeline.get_intermediate_results()
    # If this method is not implemented, show error instead and continue
    except NotImplementedError:
        print("Progress reports can only be made if the "
              "'get_intermediate_results()'-method is overridden! "
              "Skipping report.")
    # If this method is implemented, create progress report
    else:
        if mpirank != 0:
            return

        dead_samples = pipeline.intermediate_results['rejected_points']
        live_samples = pipeline.intermediate_results['live_points']
        likelihood = pipeline.intermediate_results['logLikelihood']
        lnX = pipeline.intermediate_results['lnX']

        if dead_samples is not None:
            fig = visualization.trace_plot(
                parameter_names=pipeline.parameters,
                samples=dead_samples,
                live_samples=live_samples,
                likelihood=likelihood, lnX=lnX)
            fig_filepath = os.path.join(pipeline._run_directory,
                                        'progress_report.pdf')
            msg = 'Saving progress report to {}'.format(fig_filepath)
            log.info(msg)
            fig.savefig(fig_filepath)
            if misc.is_notebook():
                ipd.clear_output()
                ipd.display(ipd.Markdown(
                    "\n**Progress report:**\nnumber of likelihood "
                    "evaluations  {}".format(
                        pipeline._likelihood_evaluations_counter)))
                plt.show()
            else:
                print(msg)


def posterior_report(pipeline, sdigits=2, **kwargs):
    """
    Displays the best fit values and 1-sigma errors for each active
    parameter. Also produces a corner plot of the samples, which is
    saved to the run directory.

    If running on a jupyter-notebook, a nice LaTeX display is used, and
    the plot is shown.

    Parameters
    ----------
    sdigits : int
        The number of significant digits to be used
    """
    out = ''
    for param, pdict in posterior_summary.items():
        if misc.is_notebook():
            # Extracts LaTeX representation from astropy unit object
            out += r"\\ \text{{ {0}: }}\; ".format(param)
            out += q2tex(*map(pdict.get, ['median', 'errup', 'errlo']),
                         sdigits=sdigits)
            out += r"\\"
        else:
            out += r"{0}: ".format(param)
            md, errlo, errup = map(pdict.get, ['median', 'errlo', 'errup'])
            if isinstance(md, apu.Quantity):
                unit = str(md.unit)
                md, errlo, errup = map(lambda x: x.value, [md, errlo, errup])
            else:
                unit = ""
            v, l, u = misc.adjust_error_intervals(
                md, errlo, errup, sdigits=sdigits)
            out += r'{0} (-{1})/(+{2}) {3}\n'.format(v, l, u, unit)

    fig = corner_plot(**kwargs)
    fig.savefig(os.path.join(pipeline._run_directory, 'corner_plot.pdf'))
    if misc.is_notebook():
        ipd.display(ipd.Markdown("\n**Posterior report:**"))
        plt.show()
        ipd.display(ipd.Math(out))
    else:
        # Restores linebreaks and prints
        print('Posterior report')
        print(out.replace(r'\n', '\n'))


def evidence_report(pipeline, sdigits=4):
    if not np.isfinite(pipeline.log_evidence):
        return

    if misc.is_notebook():
        ipd.display(ipd.Markdown("**Evidence report:**"))
        out = r"\log\mathcal{{ Z }} = "
        out += q2tex(pipeline.log_evidence, pipeline.log_evidence_err)
        ipd.display(ipd.Math(out))
    else:
        print('Evidence report')
        print('logZ =', pipeline.log_evidence, '±', pipeline.log_evidence_err)

    def prepare_likelihood_convergence_report(self, min_Nens=10, max_Nens=50,
                                              n_seeds=1, n_points=5,
                                              include_centre=True):
        """
        Constructs a report dataset based on a given Pipeline setup, which can
        be used for studying the *likelihood convergence* in a particular problem

        The pipeline's ensemble size is temporarily set to `Nens*n_seeds`, and
        (for each point) the present pipeline setup is used to compute a
        Simulations dictionary object.
        Subsets of this simulations object are then produced and the likelihood
        computed.

        The end result is a :py:obj:`pandas.DataFrame` containing the following
        columns:

          * `likelihood` - The likelihood value.
          * `likelihood_std` - The likelihood dispersion, estimated by
            bootstrapping the available ensemble and computing the standard
            deviation.
          * `ensemble_size` - Size of the ensemble of simulations used.
          * `ipoint` - Index of the point used.
          * `iseed` - Index of the random (master) seed used.
          * `param_values` - Values of the parameters at a given point.

        Parameters
        ----------
        min_Nens : int
            Minimum ensemble size to be considered
        max_Nens : int
            Maximum ensemble size to be examined
        n_seeds : int
            Number of different (master) random seeds to be used
        n_points : int
            Number of points to be evaluated. Points are randomly drawn
            from the *prior* distribution (but see `include_centre`).
        include_centre : bool
            If `True`, the first point is taken as the value corresponding to
            the centre of each parameter range.

        Returns
        -------
        results : pandas.DataFrame
            A `pandas.DataFrame` object containing the report data.
        """
        # Saves original choice for ensemble size
        original_size = self.ensemble_size
        original_likelihood_dispersion_switch = self.likelihood.compute_dispersion
        self.likelihood.compute_dispersion = True
        self.ensemble_size = max_Nens * n_seeds

        n_params = len(self.active_parameters)

        results = defaultdict(list)

        # Samples the prior distributions
        for i_point in range(n_points):
            if (i_point == 0) and include_centre:
                values = self.parameter_central_value()
                i_point = 'centre'
            else:
                # Draws a random point from the prior distributions
                values = self.prior_transform(np.random.random_sample(n_params))

            # Produces max_Nens*n_seeds outputs from this
            maps = self._get_observables(values)

            # Now, we construct smaller Simulations dicts based on a
            # subset of `maps`
            for i_seed in range(n_seeds):
                for Nens in range(min_Nens, max_Nens+1):
                    # Constructs a slice corresponding to a subset
                    indices = slice(i_seed*Nens, (i_seed+1)*Nens)
                    # Constructs the subset Simulations and computes likelihood
                    maps_subset = maps.sub_sim(indices)
                    L_value, L_std = self.likelihood(maps_subset)

                    # Stores everything
                    results['likelihood'].append(L_value)
                    results['likelihood_std'].append(L_std)
                    results['ensemble_size'].append(Nens)
                    results['ipoint'].append(i_point)
                    results['iseed'].append(i_seed)
                    results['param_values'].append(values)

        # Restores original pipeline settings
        self.ensemble_size = original_size
        self.likelihood.compute_dispersion = original_likelihood_dispersion_switch

        return results


def likelihood_convergence_report(self, cmap='cmr.chroma', **kwargs):
    """
    Prepares a standard set of plots of a likelihood convergence report
    (produced by the :py:meth:`Pipeline.prepare_likelihood_convergence_report`
    method).

    Parameters
    ----------
    cmap : str
        Colormap to be used for the lineplots
    **kwargs
        Keyword arguments that will be supplied to
        `prepare_likelihood_convergence_report` (see its docstring for
        details).
    """
    rep = self.prepare_likelihood_convergence_report(**kwargs)
    visualization.show_likelihood_convergence_report(rep, cmap)
