"""
posterior plotting with pipeline samples
"""

# visualize posterior
import numpy as np
import corner
import matplotlib
#matplotlib.use('Agg')

"""
1. load sample file
"""
samples = np.loadtxt('posterior_regular_errfix.txt')

"""
2. name of active parameters, to appear as plotting labels
"""
active_parameters = [r"$B_0$",
                     r"$\psi_0$",
                     r"$\psi_1$",
                     r"$\chi_0$",
                     r"$\alpha$",
                     r"$h_r$",
                     r"$h_z$"]

"""
3. true value of parameters
"""
truths = [6.,
          27.,
          0.9,
          25.,
          3.,
          5.,
          1.,
          0.]

"""
4. prior limit of parameters
"""
active_ranges = {r"$B_0$": (0., 10.),
                 r"$\psi_0$": (0., 50.),
                 r"$\psi_1$": (0., 2.),
                 r"$\chi_0$": (0., 50.),
                 r"$\alpha$": (1., 5.),
                 r"$h_r$": (1., 10.),
                 r"$h_z$": (0.1, 5.)}

"""
5. plotting
"""
for i in range(len(active_parameters)):  # convert variables into parameters
    low, high = active_ranges[active_parameters[i]]
    for j in range(samples.shape[0]):
        samples[j, i] = samples[j, i]*(high-low) +low
# corner plot
corner.corner(samples[:, :len(active_parameters)],
              range=[0.99] * len(active_parameters),
              bins=30,
              smooth=1.,
              smooth1d=1.,
              quantiles=[0.025, 0.16, 0.84, 0.975],
              levels=[0.68, 0.95],
              labels=active_parameters,
              show_titles=True,
              title_kwargs={"fontsize": 20},
              color='steelblue',
              truths=truths,
              truth_color='firebrick',
              plot_contours=True,
              hist_kwargs={'linewidth': 2},
              label_kwargs={'fontsize': 20})

matplotlib.pyplot.savefig('posterior_regular_errfix.pdf')
