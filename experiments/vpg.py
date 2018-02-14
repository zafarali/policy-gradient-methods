import argparse
import numpy as np
import torch.nn as nn
import torch
from pg_methods.utils.networks import MLP_factory
from pg_methods.utils.policies import MultinomialPolicy, BernoulliPolicy
from pg_methods.utils import interfaces
from pg_methods.algorithms.REINFORCE import VanillaPolicyGradient
from pg_methods.utils.baselines import MovingAverageBaseline

parser = argparse.ArgumentParser(description='REINFORCE')
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, 
                    help='random seed (default: 123)')
parser.add_argument('--n_episodes', type=int, default=5000,
                    help='number of episodes')
parser.add_argument('--n_replicates', type=int, default=4, 
                    help='number of replicates')
parser.add_argument('--baseline', type=str, default='compare',
                    help='choose one of: compare, moving_average, none')
parser.add_argument('--cpu_count', type=int, default=3,
                    help='number of cpus to use')
parser.add_argument('--plot', action='store_true', default=False)
args = parser.parse_args()

torch.manual_seed(args.seed)

EPOCHS = args.n_episodes

env = interfaces.make_parallelized_gym_env('CartPole-v0', args.seed, args.cpu_count)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

for replicate in range(args.n_replicates):

    approximator = MLP_factory(env.observation_space_info['shape'][0],
                               [32, 32],
                               output_size=env.action_space_info['possible_values'],
                               hidden_non_linearity=nn.ReLU)
    policy = MultinomialPolicy(approximator)

    optimizer = torch.optim.RMSprop(approximator.parameters(), lr=0.001)

    algorithm = VanillaPolicyGradient(env, policy, optimizer, baseline=MovingAverageBaseline(0.9))

    rewards, losses = algorithm.run(EPOCHS, verbose=True)

    np.save('rewards_vpg_{}.npy'.format(replicate), np.array(rewards))

if args.plot:
    import numpy as np
    from glob import glob
    import seaborn
    import matplotlib.pyplot as plt

    seaborn.set_color_codes('colorblind')
    seaborn.set_style('white')

    rewards = list(map(np.load, glob('./rewards_vpg_*.npy')))
    # rewards_no_baseline = list(map(np.load, glob('./rewards_no_baseline_*.npy')))


    def downsample(array, step=50):
        to_return = []
        steps = []
        for i in range(0, array.shape[0], step):
            to_return.append(array[i])
            steps.append(i)

        return np.array(steps), np.array(to_return)

    rewards = list(map(downsample, rewards))


    def plot_rewards(ax, rewards_list, label, color):
        for episode_count, reward_curve in rewards_list:
            ax.plot(episode_count, reward_curve, c=color, alpha=0.2)
        episode_count, rewards = list(zip(*rewards_list))
        episode_count = episode_count[0]
        ax.plot(episode_count, np.array(rewards).mean(axis=0), c=color,
                label=label + ' (n={})'.format(len(rewards_list)))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_rewards(ax, rewards, 'VPG', 'r')
    # plot_rewards(ax, rewards_no_baseline, 'no baseline', 'b')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title('{} - Vanilla Policy Gradient'.format(args.env_name))
    ax.legend()
    seaborn.despine()

    fig.savefig('vpg_curve.pdf', dpi=300)
