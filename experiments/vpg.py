import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pg_methods import interfaces
from pg_methods.algorithms.REINFORCE import VanillaPolicyGradient
from pg_methods.baselines import MovingAverageBaseline, NeuralNetworkBaseline
from pg_methods.networks import MLP_factory
from pg_methods.utils import experiment

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
parser.add_argument('--baseline', type=str, default='moving_average',
                    help='choose one of: none, moving_average, neural_network')
parser.add_argument('--cpu_count', type=int, default=3,
                    help='number of cpus to use')
parser.add_argument('--policy_lr', type=float, default=0.001,
                    help='learning rate for the policy')
parser.add_argument('--policy', type=str, default='multinomial',
                    help='one of (multinomial, gaussian')
parser.add_argument('--value_lr', type=float, default=0.001,
                    help='learning rate for the value function.')
parser.add_argument('--n_hidden_layers', default=1,
                    help='Number of hidden layers to use', type=int)
parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--name', default='')
args = parser.parse_args()

torch.manual_seed(args.seed)

EPOCHS = args.n_episodes

env = interfaces.make_parallelized_gym_env(args.env_name, args.seed, args.cpu_count)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

folder_name = args.env_name
algorithm_name = 'VPG' if args.name == '' else 'VPG_' + args.name

experiment_logger = experiment.Experiment({'algorithm_name': algorithm_name}, os.path.join('./', folder_name))
experiment_logger.start()

hidden_sizes = [16] * args.n_hidden_layers

for replicate in range(args.n_replicates):

    if args.baseline == 'moving_average':
        baseline = MovingAverageBaseline(0.9)
    elif args.baseline == 'neural_network':
        val_approximator = MLP_factory(env.observation_space_info['shape'][0],
                                       [16, 16],
                                       output_size=1,
                                       hidden_non_linearity=nn.ReLU)
        val_optimizer = torch.optim.SGD(val_approximator.parameters(), lr=args.value_lr)
        baseline = NeuralNetworkBaseline(val_approximator, val_optimizer, bootstrap=False)
    else:
        baseline = None

    fn_approximator, policy = experiment.setup_policy(env, hidden_non_linearity=nn.ReLU, hidden_sizes=[16, 16])

    optimizer = torch.optim.SGD(fn_approximator.parameters(), lr=args.policy_lr)

    algorithm = VanillaPolicyGradient(env, policy, optimizer, gamma=args.gamma, baseline=baseline)

    rewards, losses = algorithm.run(EPOCHS, verbose=True)

    experiment_logger.log_data('rewards', rewards.tolist())
    experiment_logger.save()


if args.plot:
    fig = plt.figure()
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.5)
    ax = fig.add_subplot(111)
    experiment_logger.plot('rewards', ax=ax)
    sns.despine()

    fig.savefig(os.path.join(experiment_logger.log_dir, experiment_logger.algorithm_details['algorithm_name']+'.pdf'))

