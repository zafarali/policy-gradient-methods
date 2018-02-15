import torch
from torch.autograd import Variable
from pg_methods.utils import interfaces

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

def calculate_bootstrapped_returns(rewards, discount, value_estimate, masks=None):
    """
    Calculates the bootstrapped return from a sequence of rewards and a final value estimate
    :param rewards: a torch.Tensor of size (time, batch_size)
    :param discount: the discount factor
    :param value_estimate: the value estimate for the last state size (1, batch_size)
    :param masks: the mask to take into account which sequences have already terminated
    :return:
    """
    value_estimate = value_estimate.view(1, -1)
    assert discount <= 1 and discount >= 0, 'discount is out of allowable range'
    if masks is None:
        # no masks at all
        masks = torch.ones_like(rewards)

    masks = masks.float()
    trajectory_length = rewards.size()[0]
    n_envs = rewards.size()[1]
    returns = torch.zeros(trajectory_length+1, *rewards.size()[1:])
    returns[-1] += value_estimate * masks[-1].view_as(value_estimate) # for those that are not masked.

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

def calculate_policy_gradient_terms(log_probs, advantage, masks=None):
    # sum over the time dimension and then mean over the batch dimension
    # to get the MC samples
    if not isinstance(log_probs, Variable):
        log_probs = Variable(log_probs)
    if not isinstance(advantage, Variable):
        advantage = Variable(advantage)

    policy_gradient_terms = -log_probs * advantage
    if masks is not None:
        policy_gradient_terms = policy_gradient_terms * Variable(masks)

    return policy_gradient_terms

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
