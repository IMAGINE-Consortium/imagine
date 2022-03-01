import nifty7 as ift
import seaborn as sns
import pylab as pl
import cmasher as cmr
# 'cmr.gothic'


def field_plotter(field_dict, cmap_string, interactive, path, plotting_kwargs=None):
    if plotting_kwargs is None:
        plotting_kwargs = dict()
    plot = ift.Plot()
    for key, field in field_dict.items():
        cmap = pl.get_cmap(cmap_string)
        if key in plotting_kwargs:
            plot.add(field, cmap=cmap, title=key, **plotting_kwargs[key])
        else:
            plot.add(field, cmap=cmap, title=key)
    if interactive:
        plot.output()
    else:
        plot.output(name=path + '.png')
        pl.close('all')


def data_plotter(field_dict, x_pos, interactive, path, uncertainty_dict=None, plotting_kwargs=None):
    if uncertainty_dict is None:
        uncertainty_dict = {}
    if plotting_kwargs is None:
        plotting_kwargs = {}
    fig, ax = pl.subplots()
    if isinstance(x_pos, ift.Field):
        x_pos = x_pos.val
    for k, v in field_dict.items():
        if isinstance(v, ift.Field):
            v = v.val
        if k in plotting_kwargs:
            ax.plot(x_pos, v, 'x', label=k, **plotting_kwargs[k])
        else:
            ax.plot(x_pos, v, 'x', label=k,)
        if k in uncertainty_dict:
            err = uncertainty_dict[k]
            if isinstance(err, ift.Field):
                err = err.val
            ax.fill_between(x_pos, v - err, v + err, alpha=0.2)
    pl.legend()
    if interactive:
        pl.show()
    else:
        pl.savefig(fname=path + '.png')
        pl.close('all')


def twod_samples(samp, a0, b0):
    def show_truth_in_jointplot(jointplot, true_x, true_y, color='r'):
        for ax in (jointplot.ax_joint, jointplot.ax_marg_x):
            ax.vlines([true_x], *ax.get_ylim(), colors=color)
        for ax in (jointplot.ax_joint, jointplot.ax_marg_y):
            ax.hlines([true_y], *ax.get_xlim(), colors=color)

    snsfig = sns.jointplot(*samp.colnames, data=samp.to_pandas(), kind='kde')
    snsfig.plot_joint(sns.scatterplot, linewidth=0, marker='.', color='0.3')
    show_truth_in_jointplot(snsfig, a0, b0)


def oned_samples(samples_dict, truths, interactive, path):
    fig, axs = pl.subplots(1, len(samples_dict), figsize=(15, 4))
    if len(samples_dict) == 1:
        k = list(samples_dict.keys())[0]
        params = list(samples_dict.values())[0]
        axs.hist(params, alpha=0.4, bins=30)
        axs.set_title(k)
        if k in truths:
            print(truths)
            axs.vlines(truths[k], ymin=0, ymax=1)
    else:
        i = 0
        for k, params in samples_dict.items():
            axs[i].hist(params, alpha=0.4, bins=30)
            axs[i].set_title(k)
            if k in truths:
                print(truths)
                axs[i].vlines(truths[k], ymin=0, ymax=1)
            i += 1
    if interactive:
        pl.show()
    else:
        pl.savefig(path + '.png')
        pl.close('all')
