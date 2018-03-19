import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import spline


sns.set_color_codes('colorblind')
sns.set_style('white')


def downsample(array, step=50):
    to_return = []
    steps = []
    for i in range(0, array.shape[0], step):
        to_return.append(array[i])
        steps.append(i)

    return np.array(steps), np.array(to_return)

def plot_lists(to_plot, ax=None, color='r', smooth=True, label='', **kwargs):
    """
    Plots a list of lists.
    This will plot each of the individual entries as a faded curve
    It will plot the mean of the curve in a solid line
    :param to_plot:
    :param ax:
    :param color:
    :param smooth:
    :param label:
    :param kwargs:
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    mean_curve = np.array(to_plot).mean(axis=0) # mean curve
    std_curve = np.array(to_plot).std(axis=0)/np.sqrt(len(to_plot)) # standard error in the mean

    x_axis = np.arange(mean_curve.shape[0])
    #
    # if smooth:
    #     x_axis
    ax.fill_between(x_axis, y1=mean_curve-std_curve,y2=mean_curve+std_curve, color=color, alpha=0.2)

    if smooth:
        downsampled_steps, downsampled_mean_curve = downsample(mean_curve)
        ax.plot(downsampled_steps, downsampled_mean_curve, c=color,
                label=label+' (n={})'.format(len(to_plot)), **kwargs)
    else:
        ax.plot(mean_curve, c=color,
                label=label + ' (n={})'.format(len(to_plot)), **kwargs)



    return ax

def plot_numbers(to_plot, ax=None, color='r', smooth=False, label=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(to_plot, c=color, label=label, **kwargs)
    return ax

