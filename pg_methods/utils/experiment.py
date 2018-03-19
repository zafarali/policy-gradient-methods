"""
Code here is used to store experimental results as well as helper functions to set up experiments
"""
from collections import defaultdict
from . import plotting
from ..networks import MLP_factory_two_heads, MLP_factory
from ..policies import GaussianPolicy, MultinomialPolicy
import os
import json

class AlgorithmDetails(object):
    """
    Example algorithm details object
    """
    def __init__(self,
                 algorithm_name,
                 learning_rate,
                 MLP_architecture):
        pass

class Experiment(object):
    def __init__(self, algorithm_details, log_dir):
        """
        Experimental utilities. Keeps track of algorithm details
        and data from replicates
        :param algorithm_details: dict with algorithm details
        :param log_dir:
        """
        self.algorithm_details = algorithm_details
        self.results = defaultdict(lambda: [])
        self.log_dir = log_dir

    def start(self):
        """
        Creates the folders and a default file
        :return:
        """
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.data_location = os.path.join(self.log_dir, '{}.json'.format(self.algorithm_details['algorithm_name']))
        json.dump({'data': {},
                   'algorithm_details': self.algorithm_details},
                  open(self.data_location, 'w'))

    def log_data(self, key, values):
        """
        Logs values under key
        :param key:
        :param values:
        :return:
        """
        self.results[key].append(values)

    def save(self):
        """
        Saves the current state of the data so far
        :return:
        """
        # TODO: check if this is efficient?
        json.dump({'data': self.results,
                   'algorithm_details': self.algorithm_details},
                  open(self.data_location, 'w'), indent=3)

    @staticmethod
    def load(log_dir):
        """
        Loads an old experiment method
        :param log_dir:
        :return:
        """
        data = json.load(open(os.path.join(log_dir), 'r'))
        exp = Experiment(data['algorithm_details'], log_dir)
        exp.results = data['data']
        return exp

    def plot(self, key, ax=None, color='red', smooth=False, label_override=False, label='', **kwargs):
        """
        Plots the data stored in `key`
        :param key:
        :param ax:
        :param color:
        :param smooth:
        :param kwargs:
        :return:
        """
        to_plot = self.results[key]

        label = self.algorithm_details['algorithm_name'] + label if not label_override else label
        if type(to_plot[0]) is list:
            ax = plotting.plot_lists(to_plot, ax=ax, color=color, smooth=smooth, label=label, **kwargs)
        else:
            ax = plotting.plot_numbers(to_plot, ax=ax, color=color, smooth=smooth, label=label, **kwargs)

        ax.set_xlabel('episodes')
        ax.set_ylabel('{}'.format(key))
        return ax


def setup_policy(env, hidden_sizes=[16], hidden_non_linearity=None):

    if env.observation_space_info['type'] == 'continuous':
        input_size = env.observation_space_info['shape'][0]
    elif env.observation_space_info['type'] == 'discrete':
        input_size = env.observation_space_info['possible_values']
    else:
        raise ValueError('Unknown observation space type {}!'.format(env.observation_space_info['type']))


    if env.action_space_info['type'] == 'continuous':
        output_size = env.action_space_info['shape'][0]
        approximator = MLP_factory_two_heads(input_size,
                                             hidden_sizes,
                                             output_size=output_size,
                                             hidden_non_linearity=hidden_non_linearity)
        policy = GaussianPolicy(approximator)

    elif env.action_space_info['type'] == 'discrete':
        output_size = env.action_space_info['possible_values']

        approximator = MLP_factory(input_size,
                                   hidden_sizes,
                                   output_size=output_size,
                                   hidden_non_linearity=hidden_non_linearity)
        policy = MultinomialPolicy(approximator)

    else:
        raise ValueError('Unknown action space type {}!'.format(env.action_space_info['type']))

    return approximator, policy