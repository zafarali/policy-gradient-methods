import torch
from pg_methods.utils import interfaces

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

#### TODO ADD TESTS:
"""
In [2]: calculate_return([0, 0, 10], 0.5)
Out[2]: [2.5, 5.0, 10.0]

In [3]: calculate_return([0, 4, 5], 0.5)
Out[3]: [3.25, 6.5, 5.0]
"""

def calculate_loss_terms(log_probs, advantage):
    """
    Returns the loss terms
    log_prob * advantage
    """
    losses = []
    loss = 0
    advantage = advantage.float()
    for t in range(len(log_probs)):
        losses.append(-log_probs[t] * advantage[0, t])
        loss += losses[-1]
    return loss, losses