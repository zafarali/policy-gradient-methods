from pg_methods.utils.data import obtain_trajectories
from pg_methods.utils.interfaces import make_parallelized_gym_env, PyTorchWrapper
import torch
from torch.autograd import Variable
import gym
import numpy as np

env = make_parallelized_gym_env('CartPole-v0', 1, 2)

def dumb_policy(state):
    return Variable(torch.IntTensor([[0], [1]])), Variable(torch.FloatTensor([[0.006], [0.001]]))

def dumb_policy_single(state):
    return Variable(torch.IntTensor([[0]])), Variable(torch.FloatTensor([[0.006]]))


def test_obtain_trajectories_max_steps():
    env.reset()

    trajectory = obtain_trajectories(env, dumb_policy, 4)
    trajectory.torchify()
    assert trajectory.rewards.size()[0] == 4, "check if step size limitation is being maintained"
    assert np.all(trajectory.dones[-1].numpy() == 0), "make sure we actually dont have any dones"

def test_obtain_trajectories_early_termination():
    env.reset()
    trajectory = obtain_trajectories(env, dumb_policy, 100)
    trajectory.torchify()
    assert torch.sum(trajectory.dones.sum(dim=1) == 1)== 1, 'Only one trajectory will terminate early'
    assert np.all(trajectory.dones[-1].numpy() == 1), 'Last row must contain only "dones"'

def test_obtain_trajectories_single_env():
    env = PyTorchWrapper(gym.make('CartPole-v0'))
    env.reset()
    trajectory = obtain_trajectories(env, dumb_policy_single, 100, verbose=True)
    trajectory.torchify()

    assert True, 'only testing if it works, not if it is correct...'