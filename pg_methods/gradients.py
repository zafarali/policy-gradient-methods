import torch
from torch.autograd import Variable
from . import interfaces

def calculate_returns(rewards, discount, masks=None):
    """
    Calculates the returns from a sequence of rewards.
    :param rewards: a torch.Tensor of size (time, batch_size)
    :param discount: a float
    :param masks: will use the mask to take into account future rewards
    """
    assert discount <= 1 and discount >= 0, 'discount is out of allowable range'
    discounted = []
    if masks is None:
        # no masks at all
        masks = torch.ones_like(rewards)
    masks = masks.float()
    trajectory_length = rewards.size()[0]
    n_processes = rewards.size()[1]
    returns = torch.zeros(trajectory_length+1, *rewards.size()[1:])
    for t in reversed(range(trajectory_length)):
        returns[t] = discount * returns[t+1] + masks[t] * rewards[t]

    return returns[:-1]

def calculate_advantages(rewards, discount, baselines=None):
    """
    Calculates the returns/advantages of the rewards
    :param rewards: a list of rewards
    :param discount: the discount factor
    :param baselines: the baseline of states corresponding to rewards.
    :return list: containing advantages for each step in the trajectory
    """
    # rewards = interfaces.pytorch2list(rewards)
    # baselines = interfaces.pytorch2list(baselines)
    assert discount < 1 and discount >= 0, 'discount is out of allowable range'
    advantage = []
    trajectory_length = len(rewards)
    
    if baselines is None: baselines = [0]*trajectory_length

    return_t = 0
    
    # go backwards in time to calculate this
    for t in reversed(range(trajectory_length)):
        return_t = discount * return_t + rewards[t]
        advantage.insert(0, return_t-baselines[t])

    return interfaces.list2pytorch(advantage)


def calculate_gae(rewards, values, discount, tau, masks=None):
    """
    Calculate the generalized advantages of the rewards
    :param rewards: a list of rewards
    :param values: a list of returns approximated using value function at each state
    :param discount: discount factor
    :param tau: lambda parameter for generalized advantage estimation
    :param masks: will use the mask to take into account future rewards
    :return list: containing generalized advantages for each step in the trajectory
    """
    assert discount <= 1 and discount >= 0, 'discount is out of allowable range'
    assert tau <= 1 and tau >= 0, 'bias control(tau) is out of allowable range'
    trajectory_length = len(rewards) - 1
    advantages = []

    gae = 0

    for t in reversed(range(trajectory_length)):
        delta_t = rewards[t] + discount * values[t+1] - values[t]
        gae = gae * discount * tau + delta_t

        advantages.insert(0, gae)

    return interfaces.list2pytorch(advantages)


def calculate_policy_gradient_terms(log_probs, advantage):
    # sum over the time dimension and then mean over the batch dimension
    # to get the MC samples
    if not isinstance(log_probs, Variable):
        log_probs = Variable(log_probs)
    if not isinstance(advantage, Variable):
        advantage = Variable(advantage)

    return -log_probs.float() * advantage.float()

def get_entropy(log_probs):
    """
    Calculates the entropy of a distribution
    :param log_probs: the log probabilities output by the policy
    :return: the entropy
    """
    entropy = -log_probs.exp() * log_probs
    if not isinstance(entropy, Variable):
        entropy = Variable(entropy)
    return entropy
