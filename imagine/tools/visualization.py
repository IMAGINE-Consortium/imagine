"""
This module contains convenient standard plotting functions
"""
# %% IMPORTS
# Built-in imports
from copy import copy

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import cmasher as cmr
import healpy as hp

def corner_plot(pipeline=None, truths_dict=None, show_sigma=True, param_names=None,
                table=None, samples=None, live_samples=None,
                **kwargs):
    """
    Makes a corner plot.

    If a :py:obj:`Pipeline <imagine.pipelines.pipeline.Pipeline>` object is
    supplied, it will be used to collect all the necessary information.
    Alternatively, one can supply either a :py:obj:`astropy.table.Table`
    or a :py:obj:`numpy.ndarray` containing with different parameters as
    columns.

    The plotting is done using the :py:mod:`corner` package, and extra
    keyword parameters are passed directly to it

    Parameters
    ----------
    pipeline : imagine.pipelines.pipeline.Pipeline
        Pipeline from which samples are read in the default case.
    truths_dict : dict
        Dictionary containing active parameters as keys and the expected values
        as values
    show_sigma : bool
        If True, plots the 1, 2 and 3-sigma contours.
    param_names : list
        If present, only parameters from this list will be plotted
    table : astropy.Table
        If present, samples from this table are used instead of the Pipeline.
    samples : numpy.ndarray
        If present, samples are read from this array
    live_samples : numpy.ndarray
        If this array is present, a second set of samples are shown in the
        plots.

    Returns
    -------
    corner_fig : matplotlib.Figure
        Figure containing the generated corner plot
    """

    if samples is None:
        if table is None:
            samp = pipeline.samples
        else:
            samp = table
        if param_names is None:
            param_names = list(samp.columns)
        samples = np.vstack([samp[i].value for i in param_names]).T

    if truths_dict is None:
        truths_list = None
    else:
        truths_list = []
        for p in param_names:
            if p in truths_dict:
                truths_list.append(truths_dict[p])
            else:
                truths_list.append(None)

    # 1, 2, 3-sigma levels
    levels=[1-np.exp(-0.5*i**2) for i in range(1,4)]

    config = dict(color='tab:blue', truths=truths_list,
                  labels=param_names,
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
                                figsize=(8.3, height), dpi=125)
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

            ax.plot(edges[:-1], hist, color=colors[i], drawstyle='steps-pre',
                    label='rej.')

            if live_samples is not None:
                hist_live, edges_live = np.histogram(live_samples[:,i-1],
                                                     bins=hist_bins)

                # Makes sure everything is visible in the same histogram
                hist_live = hist_live* hist.max()/hist_live.max()

                ax.plot(edges_live[:-1], hist_live, color=color_live,
                        drawstyle='steps-pre', label='live')
            ax.legend(frameon=False)

    plt.tight_layout()
    return fig

__divergent_quantitites = {'fd'}

def _choose_cmap(title=None):
    """
    Chooses a divergent colormap for signed quantities
    """
    if title is not None:
        # Takes only the observable name
        title = title.split()[0].strip()
    if title in __divergent_quantitites:
        cmap = 'cmr.fusion'
    else:
        cmap = 'cmr.rainforest'
    return copy(plt.get_cmap(cmap))

def _key_formatter(key):
    """
    Converts a ObservableDict key into a plot title
    """
    name, freq, Nside, tag = key

    if freq is not None:
        freq = '  {} GHz'.format(freq)
    else:
        freq = ''
    if tag is None:
        tag = ''

    return '{name} {tag} {freq}'.format(name=name, tag=tag, freq=freq)

def show_observable(obs, realization=0, title=None, cartesian_axes='yz',
                    is_covariance=False, **kwargs):
    """
    Displays the contents of a single realisation of an
    Observable object.

    Parameters
    ----------
    obs : imagine.observables.observable.Observable
        Observable object whose contents one wants to plot
    realization : int
        Index of the ensemble realization to be plotted
    cartesian_axes : str
        If plotting a tabular observable using cartesian coordinates,
        this allows selecting which two axes should be used for the plot.
        E.g. 'xy', 'zy', 'xz'. Default: 'yz'.
    **kwargs
        Parameters to be passed to the apropriate plotting routine
        (either `healpy.visufunc.mollview` or `matplotlib.pyplot.imshow`).
    """
    if obs.otype == 'HEALPix':
        default_cmap = _choose_cmap(title=title)
        mollview_args = {'norm': 'hist',
                         'cmap': copy(default_cmap),
                         'unit': obs.unit._repr_latex_()}
        mollview_args.update(kwargs)
        return hp.mollview(obs.global_data[realization], title=title,
                           **mollview_args)

    elif obs.otype == 'tabular':
        if 'sub' in kwargs:
            ax = plt.subplot(*kwargs['sub'])
        else:
            ax = plt.gca()

        if obs.coords['type'] == 'galactic':
            x, y = obs.coords['lon'], obs.coords['lat']
            ax.set_xlabel('Gal. lon. [{}]'.format(x.unit._repr_latex_()))
            ax.set_ylabel('Gal. lat. [{}]'.format(y.unit._repr_latex_()))
        elif obs.coords['type'] == 'cartesian':
            x, y = obs.coords[cartesian_axes[0]], obs.coords[cartesian_axes[1]]
            ax.set_xlabel('{} [{}]'.format(cartesian_axes[0], x.unit._repr_latex_()))
            ax.set_ylabel('{} [{}]'.format(cartesian_axes[1], y.unit._repr_latex_()))
        else:
            raise ValueError('Unsupported coordinates type', obs.coords['type'])

        values = obs.global_data[realization]

        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
        else:
            cmap = 'cmr.chroma'
        cmap = cmr.get_sub_cmap(cmap, 0.15, 0.85)

        ax.set_title(title)
        im = ax.scatter(x, y, c=values, cmap=cmap)
        ax.grid(alpha=0.2)
        plt.colorbar(im, ax=ax, label=obs.unit._repr_latex_())

    elif is_covariance:
        if 'sub' in kwargs:
            ax = plt.subplot(*kwargs['sub'])
        else:
            ax = plt.gca()
        ax.set_title(title)
        # Makes the color interval symmetric
        vmax = np.abs([obs.global_data.min(), obs.global_data.max()]).max()
        im = ax.imshow(obs.global_data, cmap='cmr.fusion',
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label=obs.unit._repr_latex_())
    else:
        # Includes the title in the corresponding subplot, even in
        # the unsupported case (so that the user remembers it)
        print("Plotting observable type '{}' not yet implemented".format(obs.otype))

def show_observable_dict(obs_dict, max_realizations=None, **kwargs):
    """
    Plots the contents of an ObservableDict object.

    Parameters
    ----------
    obs_dict : imagine.observables.observable.ObservableDict
        ObservableDict object whose contents one wants to plot.
    max_realization : int
        Index of the maximum ensemble realization to be plotted. If None,
        the whole ensemble is shown.
    **kwargs
        Parameters to be passed to the apropriate plotting routine
        (either :py:func:`healpy.visufunc.mollview` or :py:func:`matplotlib.pyplot.imshow`).
    """
    # This import needs to be inside the function to avoid problems
    from imagine.observables import Covariances, ObservableDict

    assert isinstance(obs_dict, ObservableDict)

    keys = list(obs_dict.keys())
    ncols = len(keys)

    is_covariance = isinstance(obs_dict, Covariances)
    if ncols == 0:
        print('The ObservableDict is empty.')
        return

    if is_covariance:
        nrows = 1
    else:
        # Gets the ensemble size from the first Observable
        # (each realisation corresponds to a row)
        nrows = obs_dict[keys[0]].shape[0]
        if max_realizations is not None:
            nrows = min(nrows, max_realizations)

    i_subplot = 0
    for j in range(nrows):
        for i, key in enumerate(keys):
            i_subplot += 1
            title = _key_formatter(key)
            show_observable(obs_dict[key], title=title, realization=j,
                            is_covariance=is_covariance,
                            sub=(nrows, ncols, i_subplot), **kwargs)

