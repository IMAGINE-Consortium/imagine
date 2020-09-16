"""
This module contains convenient standard plotting functions
"""
import numpy as np
from corner import corner

def corner_plot(pipeline=None, samples=None, live_samples=None, truths_dict=None,
               show_sigma=True, **kwargs):


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
