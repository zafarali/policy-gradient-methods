import matplotlib
matplotlib.use('Agg')
import argparse
from glob import glob
import os
import matplotlib.pyplot as plt
import math
import gc
import seaborn as sns
from pg_methods.utils import experiment
sns.set_style('white')
sns.set_color_codes('colorblind')
sns.set_context('paper', font_scale=1)
parser = argparse.ArgumentParser(description='Experiment plotter')
parser.add_argument('--name', default='plots', help='Output file appenditure')
parser.add_argument('--to_plot', default='rewards', help='The thing to plot on the y-axis')
parser.add_argument('--folder', default='./', help='Name of folder to read data from')
parser.add_argument('--figx', default=None)
parser.add_argument('--figy', default=None)
args = parser.parse_args()

environments = glob(os.path.join(args.folder, '*-v*'))

if args.figx is not None and args.figy is not None:
    fig = plt.figure(figsize=(args.figx, args.figy))
else:
    fig = plt.figure()

COLORS = sns.color_palette('colorblind')
print('Number of environments:', len(environments))
for i, environment in enumerate(environments):
    ax = fig.add_subplot(3, math.ceil(len(environments) / 3)+1, i+1)
    algorithms = sorted(glob(os.path.join(environment, '*.json')))
    for color, algorithm in zip(COLORS, algorithms):
        data = experiment.Experiment.load(algorithm)
        try:
            data.plot(args.to_plot, ax=ax, color=color, smooth=True)
        except KeyError as e:
            print('Key not found: {} in algorithm {}'.format(e, algorithm))
    ax.set_title(os.path.basename(environment))
    ax.legend(fontsize=6, loc='center right')
fig.tight_layout()
fig.savefig(os.path.join(args.folder, args.to_plot+'-'+args.name+'.pdf'))
fig.clf()
plt.close()
gc.collect()