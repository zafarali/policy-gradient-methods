"""
Code to compare REINFORCE with and without baselines
"""
import gym 
import argparse
import numpy as np
import torch.nn as nn
import torch
from pg_methods.utils.approximators import MLP_factory
from pg_methods.utils.policies import MultinomialPolicy, BernoulliPolicy
from pg_methods.utils import interfaces
from pg_methods.algorithms.REINFORCE import REINFORCE
from pg_methods.utils.baselines import MovingAverageBaseline


parser = argparse.ArgumentParser(description='REINFORCE')
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, 
                    help='random seed (default: 123)')
parser.add_argument('--n_episodes', type=int, default=10000,
                    help='number of episodes')
parser.add_argument('--n_replicates', type=int, default=4, 
                    help='number of replicates')
parser.add_argument('--baseline', type=str, default='compare',
                    help='choose one of: compare, moving_average, none')
args = parser.parse_args()

torch.manual_seed(args.seed)

EPOCHS = args.n_episodes

if args.baseline == 'compare':
    baselines = [MovingAverageBaseline(0.8), None]
elif args.baseline == 'moving_average':
    baselines = [MovingAverageBaseline(0.8)]
else:
    baselines = [None]

environment = gym.make('CartPole-v0')
state_processor = interfaces.SimpleStateProcessor(
                                environment.observation_space,
                                one_hot=False)

action_processor = interfaces.SimpleActionProcessor(environment.action_space)

for baseline in baselines:
    print('BASELINE: {}'.format(baseline))
    for replicate in range(args.n_replicates):
        print('REPLICATE #: {}'.format(replicate))
        approximator = MLP_factory(state_processor.state_size,
                                   [32, 32],
                                   output_size=action_processor.action_space.n,
                                   hidden_non_linearity=nn.ReLU)
        policy = MultinomialPolicy(approximator)

        optimizer = torch.optim.Adam(approximator.parameters(), lr=0.00005)

        algorithm = REINFORCE(environment,
                              policy,
                              optimizer,
                              state_processor,
                              action_processor,
                              gamma=0.99,
                              baseline=baseline)
        try:
            rewards, losses = algorithm.run(EPOCHS, verbose=True)
        except KeyboardInterrupt as e:
            print('Training stopped early')
        
        if baseline is None:
            np.save('rewards_no_baseline_{}.npy'.format(replicate), np.array(rewards))
        else:
            np.save('rewards_with_baseline_{}.npy'.format(replicate), np.array(rewards))

"""
Code functionality for plotting
"""
if args.baseline == 'compare':
    import numpy as np
    from glob import glob 
    import seaborn
    import matplotlib.pyplot as plt
    seaborn.set_color_codes('colorblind')
    seaborn.set_style('white')

    rewards_baseline = list(map(np.load, glob('./rewards_with_baseline_*.npy')))
    rewards_no_baseline = list(map(np.load, glob('./rewards_no_baseline_*.npy')))

    def downsample(array, step=50):
        to_return = []
        steps = []
        for i in range(0, array.shape[0], step):
            to_return.append(array[i])
            steps.append(i)
        
        return np.array(steps), np.array(to_return)

    rewards_baseline = list(map(downsample, rewards_baseline))
    rewards_no_baseline = list(map(downsample, rewards_no_baseline))

    def plot_rewards(ax, rewards_list, label, color):
        for episode_count, reward_curve in rewards_list:
            ax.plot(episode_count, reward_curve, c=color, alpha=0.2)
        episode_count, rewards = list(zip(*rewards_list))
        episode_count = episode_count[0]
        ax.plot(episode_count, np.array(rewards).mean(axis=0), c=color, label=label+' (n={})'.format(len(rewards_list)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_rewards(ax, rewards_baseline, 'with baseline', 'r')
    plot_rewards(ax, rewards_no_baseline, 'no baseline', 'b')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title('{}'.format(args.env_name))
    ax.legend()
    seaborn.despine()

    fig.savefig('reward_curves_comparison.pdf', dpi=300)
