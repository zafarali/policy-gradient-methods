"""
Here we implement some simple policies that 
one can use directly in simple tasks. 
More complicated policies can also be created
by inheriting from the Policy class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from torch.distributions import Categorical, Normal, Bernoulli

class Policy(nn.Module):
    def __init__(self, fn_approximator):
        super().__init__()
        self.fn_approximator = fn_approximator
    def forward(self, state):
        raise NotImplementedError('Must be implemented.')

class RandomPolicy(Policy):
    """
    A random policy that just takes one of output_dim actions randomly
    """
    def __init__(self, output_dim=2):
        super().__init__(None)
        self.output_dim = output_dim
        self.p = nn.Parameter(torch.IntTensor([0]), requires_grad=False)


    def forward(self, state):
        batch_size = state.size()[0]
        probs = Variable(torch.ones(batch_size, self.output_dim)/self.output_dim)
        stochastic_policy = Categorical(probs)
        actions = stochastic_policy.sample()
        log_probs = stochastic_policy.log_prob(actions)
        return actions, log_probs


class MultinomialPolicy(Policy):
    """
    Used to pick from a range of actions.
    ```
    fn_approximator = MLP_factory(input_size=4, output_size=3)
    policy = policies.MultinomialPolicy(fn_approximator)
    the actions will be a number in [0, 1, 2]
    ```
    """
    def forward(self, state):
        policy_log_probs = self.fn_approximator(state) 
        probs = F.softmax(policy_log_probs, dim=1)
        stochastic_policy = Categorical(probs)

        # sample discrete actions
        actions = stochastic_policy.sample()

        # get log probs
        log_probs = stochastic_policy.log_prob(actions)

        return actions, log_probs

    def log_prob(self, state, action):
        policy_log_probs = self.fn_approximator(state)
        probs = F.softmax(policy_log_probs, dim=1)
        stochastic_policy = Categorical(probs)
        return stochastic_policy.log_prob(action)

class GaussianPolicy(Policy):
    """
    Used to take actions in continous spaces
    ```
    fn_approximator = MLP_factory(input_size=4, output_size=2)
    policy = policies.GaussianPolicy(fn_approximator)
    ```
    """
    def forward(self, state):
        nn_output = self.fn_approximator(state)
        policy_mu, policy_sigma = nn_output[:, 0], nn_output[:, 1]
        policy_sigma = F.softplus(policy_sigma)

        stochastic_policy = Normal(policy_mu, policy_sigma)

        actions = stochastic_policy.sample()

        log_probs = stochastic_policy.log_prob(actions)

        return actions, log_probs

    def log_prob(self, state, action):
        raise NotImplementedError('Not implemented yet')

class BernoulliPolicy(Policy):
    """
    Used to take binary actions. 
    This can also be used when each action consists of
    a many binary actions, for example:
    
    ```
    fn_approximator = MLP_factory(input_size=4, output_size=5)
    policy = policies.BernoulliPolicy(fn_approximator)
    ```    
    this will result in each action being composed of 5 binary actions.
    """
    def forward(self, state):
        policy_p = self.fn_approximator(state)
        policy_p = F.sigmoid(policy_p)

        try:
            stochastic_policy = Bernoulli(policy_p)

            actions = stochastic_policy.sample()

            log_probs = stochastic_policy.log_prob(actions)
        except RuntimeError as e:
            logging.debug('Runtime error occured. policy_p was {}'.format(policy_p))
            logging.debug('State was: {}'.format(state))
            logging.debug('Function approximator return was: {}'.format(self.fn_approximator(state)))
            logging.debug('This has occured before when parameters of the network became NaNs.')
            logging.debug('Check learning rate, or change eps in adaptive gradient descent methods.')
            raise RuntimeError('BernoulliPolicy returned nan information. Logger level with DEBUG will have more '
                               'information')
        return actions, log_probs

    def log_prob(self, state, action):
        policy_p = self.fn_approximator(state)
        policy_p = F.sigmoid(policy_p)
        stochastic_policy = Bernoulli(policy_p)
        return stochastic_policy.log_prob(action)
