"""
This module contains convenient standard plotting functions
"""
import numpy as np
from corner import corner

def corner_plot(pipeline=None, samples=None, live_samples=None, truths=None,
               show_sigma=True, **kwargs):


    if truths is not None:
        truths_list = []
        for p in pipeline.active_parameters:
            if p in truths:
                truths_list.append(truths[p])
            else:
                truths_list.append(None)
        truths = truths_list

    if samples is None:
        samp = pipeline.samples
        samples = np.vstack([samp[i].value
                             for i in pipeline.active_parameters]).T

    # 1, 2, 3-sigma levels
    levels=[1-np.exp(-0.5*i**2) for i in range(1,4)]

    corner_fig = corner(samples, truths=truths, color='tab:blue',
                        labels=pipeline.active_parameters,
                        fill_contours=False, levels=levels,
                        plot_contours=show_sigma,
                        show_titles=True,
                        quantiles=[0.022, 0.5, 0.977],
                        label_kwargs={'fontsize': 10},
                        **kwargs)

    if live_samples is not None:
        corner_fig.axes[2].scatter(live_samples[:,0],live_samples[:,1],
                                   marker='.', c='tab:orange', s=1, alpha=0.5)
        corner_fig.axes[0].hist(live_samples[:,0], color='tab:orange', alpha=0.5)
        corner_fig.axes[3].hist(live_samples[:,1], color='tab:orange', alpha=0.5)

    return corner_fig
