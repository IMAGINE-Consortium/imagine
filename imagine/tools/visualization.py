"""
This module contains convenient standard plotting functions
"""
import numpy as np
from corner import corner

def cornerplot(samples, live_samples=None, pipeline=None,
               truths=None, show_sigma=True):

    # 1, 2, 3-sigma levels
    levels=[1-np.exp(-0.5*i**2) for i in range(1,4)]

    corner_fig = corner(samples, truths=truths, color='tab:blue',
                        labels=pipeline.active_parameters,
                        range=[0.8]*len(pipeline.active_parameters),
                        fill_contours=False, levels=levels,
                        plot_contours=show_sigma,
                        show_titles=True,
                        quantiles=[0.022, 0.5, 0.977],
                        label_kwargs={'fontsize': 10})

    if live_samples is not None:
        corner_fig.axes[2].scatter(live_samples[:,0],live_samples[:,1],
                                   marker='.', c='tab:orange', s=1, alpha=0.5)
        corner_fig.axes[0].hist(live_samples[:,0], color='tab:orange', alpha=0.5)
        corner_fig.axes[3].hist(live_samples[:,1], color='tab:orange', alpha=0.5);
    return corner_fig
