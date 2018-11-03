"""
Test the VPG algorithm, also serves as an integration test.
"""

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

def setup_env(env_name='CartPole-v0'):
    env = interfaces.make_parallelized_gym_env(env_name,
                                               seed=5,
                                               n_workers=2)
    return env

def setup_policy(env):
    torch.manual_seed(0)
    np.random.seed(0)

    experiment_logger = experiment.Experiment(
        {'algorithm_name': 'VPG'}, './tmp/')
    experiment_logger.start()

    fn_approximator, policy = experiment.setup_policy(env,
                                                      hidden_non_linearity=nn.ReLU,
                                                      hidden_sizes=[16, 16])
    optimizer = torch.optim.SGD(fn_approximator.parameters(), lr=0.01)
    return fn_approximator, policy, optimizer

def setup_baseline(baseline_type, env=None):
    if baseline_type == 'moving_averag':
        return MovingAverageBaseline(0.9)
    elif baseline_type == 'neural_network':
        val_approximator = MLP_factory(env.observation_space_info['shape'][0],
                                       [16, 16],
                                       output_size=1,
                                       hidden_non_linearity=nn.ReLU)
        val_optimizer = torch.optim.SGD(val_approximator.parameters(), lr=0.001)
        return NeuralNetworkBaseline(val_approximator, val_optimizer, bootstrap=False)
    else:
        return None

def run_algorithm(env, policy, optimizer, baseline):
    algorithm = VanillaPolicyGradient(env, policy, optimizer,
                                      gamma=0.99, baseline=baseline)
    algorithm.run(5, verbose=True)
    return True

def test_vpg_moving_average_baseline():
    env = setup_env()
    fn_approximator, policy, optimizer = setup_policy(env)
    baseline = setup_baseline('moving_average', env)
    run_algorithm(env, policy, optimizer, baseline)

def test_vpg_neuralnetwork_baseline():
    env = setup_env()
    fn_approximator, policy, optimizer = setup_policy(env)
    baseline = setup_baseline('neural_network', env)
    run_algorithm(env, policy, optimizer, baseline)


def test_vpg_nobaseline():
    env = setup_env()
    fn_approximator, policy, optimizer = setup_policy(env)
    baseline = setup_baseline('none', env)
    run_algorithm(env, policy, optimizer, baseline)

