import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical, Normal, Bernoulli

class Policy(nn.Module):
    def __init__(self, fn_approximator):
        super().__init__()
        self.fn_approximator = fn_approximator
    def forward(self, state):
        raise NotImplementedError('Must be implemented.')

class MultinomialPolicy(Policy):
    """
    Used to pick from a range of actions
    """
    def forward(self, state):
        policy_log_probs = self.fn_approximator(state) 
        probs = F.softmax(policy_log_probs)
        stochastic_policy = Categorical(probs)

        # sample discrete actions
        actions = stochastic_policy.sample()

        # get log probs
        log_probs = stochastic_policy.log_prob(actions)

        return actions, log_probs

    def log_prob(self, state, action):
        policy_log_probs = self.fn_approximator(state)
        probs = F.softmax(policy_log_probs)
        stochastic_policy = Categorical(probs)
        return stochastic_policy.log_prob(action)

class GaussianPolicy(Policy):
    """
    Used to take actions in continous spaces
    """
    def forward(self, state):
        policy_mu, policy_sigma = self.fn_approximator(state)
        policy_sigma = F.softplus(policy_sigma)

        stochastic_policy = Normal(policy_mu, policy_sigma)

        actions = stochastic_policy.sample()

        log_probs = stochastic_policy.log_prob(actions)

        return actions, log_probs

    def log_prob(self, state, action):
        raise NotImplementedError('Not implemented yet')

class BernoulliPolicy(Policy):
    """
    Used to take binary actions
    """
    def forward(self, state):
        policy_p = self.fn_approximator(state)
        policy_p = F.sigmoid(policy_p)

        try:
            stochastic_policy = Bernoulli(policy_p)

            actions = stochastic_policy.sample()

            log_probs = stochastic_policy.log_prob(actions)
        except RuntimeError as e:
            print('Runtime error occured. policy_p was {}'.format(policy_p))
        return actions, log_probs

    def log_prob(self, state, action):
        policy_p = self.fn_approximator(state)
        policy_p = F.sigmoid(policy_p)
        stochastic_policy = Bernoulli(policy_p)
        return stochastic_policy.log_prob(action)