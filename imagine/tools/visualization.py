"""
This module contains convenient standard plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import cmasher as cmr

def corner_plot(pipeline=None, samples=None, live_samples=None, truths_dict=None,
                show_sigma=True, **kwargs):
    """

    """
    if truths_dict is None:
        truths_list = None
    else:
        truths_list = []
        for p in pipeline.active_parameters:
            if p in truths_dict:
                truths_list.append(truths_dict[p])
            else:
                truths_list.append(None)

    if samples is None:
        samp = pipeline.samples
        samples = np.vstack([samp[i].value
                             for i in pipeline.active_parameters]).T

    # 1, 2, 3-sigma levels
    levels=[1-np.exp(-0.5*i**2) for i in range(1,4)]

    config = dict(color='tab:blue', truths=truths_list,
                  labels=pipeline.active_parameters,
                  fill_contours=False,
                  levels=levels,
                  plot_contours=show_sigma,
                  show_titles=True,
                  quantiles=[0.022, 0.5, 0.977],
                  truth_color='darkred',
                  label_kwargs={'fontsize': 10})
    config.update(kwargs)  # Allows overriding defaults
    corner_fig = corner(samples, **config)

    if live_samples is not None:
        corner_fig.axes[2].scatter(live_samples[:,0],live_samples[:,1],
                                   marker='.', c='tab:orange', s=1, alpha=0.5)
        corner_fig.axes[0].hist(live_samples[:,0], color='tab:orange', alpha=0.5)
        corner_fig.axes[3].hist(live_samples[:,1], color='tab:orange', alpha=0.5)

    return corner_fig

def trace_plot(samples=None, live_samples=None, likelihood=None,
               lnX=None, parameter_names=None, cmap='cmr.ocean',
               color_live='#e34a33', fig=None, hist_bins=30):
    """
    Produces a set of "trace plots" for a nested sampling run,
    showing the position of "dead" points as a function of prior mass.
    Also plots the distributions of dead points accumulated until now, and the
    distributions of live points.

    Parameters
    ----------
    samples : numpy.ndarray
        (Nsamples, Npars)-array containing the rejected points
    likelihood : numpy.ndarray
        Nsamples-array containing the log likelihood values
    lnX : numpy.ndarray
        Nsamples-array containing the "prior mass"
    parameter_names : list or tuple
        List of the nPars active parameter names
    live_samples : numpy.ndarray, optional
        (Nsamples, Npars)-array containing the present live points
    cmap : str
        Name of the colormap to be used
    color_live : str
        Colour used for the live points distributions (if those are present)
    fig : matplotlib.Figure
        If a previous figure was generated, it can be passed to this function
        for update using this argument
    hist_bins : int
        The number of bins used for the histograms

    Returns
    -------
    fig : matplotlib.Figure
        The figure produced
    """
    _, nParams = samples.shape

    nrows = nParams+1
    height = min(nrows*1.5, 11.7) # Less than A4 height

    if fig is None:
        fig, axs = plt.subplots(nrows=nrows, ncols=2,
                                gridspec_kw={'width_ratios':[3,2]},
                                figsize=(8.3, height), dpi=200)
    else:
        axs = np.array(fig.axes).reshape(nrows,2)

    norm_likelihood = likelihood
    norm_likelihood = norm_likelihood - norm_likelihood.min()
    norm_likelihood = norm_likelihood/norm_likelihood.max()

    plot_settings = {'marker': '.', 'linestyle':'none'}

    colors = cmr.take_cmap_colors(cmap, nParams+1,
                                  cmap_range=(0.1, 0.85),
                                  return_fmt='hex')

    # Works on the trace plots
    for i, ax in enumerate(axs[:,0]):
        if i==0:
            y = norm_likelihood
            ax.set_ylabel(r'$\ln\mathcal{L}$'+'\n(normaliz.)')
        else:
            y = samples[:,i-1]
            if parameter_names is not None:
                ax.set_ylabel(parameter_names[i-1])
            else:
                ax.set_ylabel(i)

        plot_settings['color'] = colors[i]
        ax.plot(-lnX, y, **plot_settings)
        ax.set_xlabel('$-\ln X$')

    # Works on the histograms
    for i, ax in enumerate(axs[:,1]):
        if i==0:
            ax.set_axis_off()
        else:
            if parameter_names is not None:
                ax.set_xlabel(parameter_names[i-1])
            else:
                ax.set_xlabel(i)
            hist, edges = np.histogram(samples[:,i-1], bins=hist_bins)

            ax.plot(edges[:-1], hist, color=colors[i], drawstyle='steps-pre')

            if live_samples is not None:
                hist_live, edges_live = np.histogram(live_samples[:,i-1],
                                                     bins=hist_bins)

                # Makes sure everything is visible in the same histogram
                hist_live = hist_live* hist.max()/hist_live.max()

                ax.plot(edges_live[:-1], hist_live, color=color_live, drawstyle='steps-pre')


    plt.tight_layout()
    return fig
