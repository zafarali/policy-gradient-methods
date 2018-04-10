import numpy as np
import torch
from torch.autograd import Variable
import pg_methods.policies as policies
from pg_methods.networks import MLP_factory, MLP_factory_two_heads

def test_categorial_policy():
    fn_approximator = MLP_factory(input_size=4, output_size=3)
    policy = policies.CategoricalPolicy(fn_approximator)

    action, log_prob = policy(Variable(torch.randn(1, 4), volatile=True))
    assert type(action.data[0]) is int
    assert log_prob.data[0] <= np.log(1)

def test_categorial_policy_batched():
    fn_approximator = MLP_factory(input_size=4, output_size=3)
    policy = policies.CategoricalPolicy(fn_approximator)

    action, log_prob = policy(Variable(torch.randn(10, 4), volatile=True))

    assert tuple(action.size()) == (10, ) # we must get back 10 actions.
    assert sum([type(action.data[i]) is int for i in range(action.size()[0])])
    assert torch.sum(log_prob.data <= torch.log(torch.ones_like(log_prob.data)))

def test_gaussian_policy():
    fn_approximator = MLP_factory_two_heads(input_size=4, hidden_sizes=[16], output_size=2)
    policy = policies.GaussianPolicy(fn_approximator)

    action, log_prob = policy(Variable(torch.randn(1, 4), volatile=True))

    assert type(action.data[0][0]) is float
    assert log_prob.data[0][0] <= np.log(1)

def test_gaussian_policy_batched():
    fn_approximator = MLP_factory_two_heads(input_size=4, hidden_sizes=[16], output_size=1)
    policy = policies.GaussianPolicy(fn_approximator)

    action, log_prob = policy(Variable(torch.randn(10, 4), volatile=True))
    print(action.size())

    assert tuple(action.size()) == (10, 1) # we must get back 10 actions.
    assert sum([type(action.data[i][0]) is float for i in range(action.size()[0])])
    assert torch.sum(log_prob.data <= torch.log(torch.ones_like(log_prob.data)))

def test_bernoulli_policy_batched():
    fn_approximator = MLP_factory(input_size=4, output_size=1)
    policy = policies.BernoulliPolicy(fn_approximator)

    action, log_prob = policy(Variable(torch.randn(10, 4), volatile=True))
    assert tuple(action.size()) == (10, 1) # we must get back 10 actions.
    assert torch.sum(log_prob.data <= torch.log(torch.ones_like(log_prob.data)))
    assert sum([action.data[i, 0] in [0, 1] for i in range(action.size()[0])]), 'Actions are not between 0 and 1'

def test_multi_bernoulli_policy_batched():
    """
    Simulates when each action consists of 5 bernoulli choices.
    """
    fn_approximator = MLP_factory(input_size=4, output_size=5)
    policy = policies.BernoulliPolicy(fn_approximator)

    action, log_prob = policy(Variable(torch.randn(10, 4), volatile=True))

    assert tuple(action.size()) == (10, 5) # we must get back 10 actions with 5 switches.
    assert torch.sum(log_prob.data <= torch.log(torch.ones_like(log_prob.data)))
    assert sum([action.data[i, 0] in [0, 1] for i in range(action.size()[0])]), 'Actions are not between 0 and 1'

