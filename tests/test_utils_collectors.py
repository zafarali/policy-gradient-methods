from pg_methods.data import obtain_trajectories
from pg_methods.data import MultiTrajectory
from pg_methods.interfaces import (make_parallelized_gym_env,
                                   PyTorchWrapper,
                                   SimpleDiscreteProcessor,
                                   ContinuousProcessor)
from pg_methods.gradients import calculate_returns
import torch
from torch.autograd import Variable
import gym
import numpy as np

env = make_parallelized_gym_env('CartPole-v0', 1, 2,
                                observation_processor=ContinuousProcessor(),
                                action_processor=SimpleDiscreteProcessor())

def dumb_policy(state):
    return Variable(torch.IntTensor([[0], [1]])), Variable(torch.FloatTensor([[0.006], [0.001]]))

def dumb_policy_single(state):
    return Variable(torch.IntTensor([[0]])), Variable(torch.FloatTensor([[0.006]]))


def test_obtain_trajectories_max_steps():
    env.reset()

    trajectory = obtain_trajectories(env, dumb_policy, 4)
    print(trajectory.states)
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

# TODO: add a test for mask-correction check
def test_mask_correction():
    traj = MultiTrajectory(2)

    state_t = torch.FloatTensor([[1, 1], [1, 1]])
    state_tp1 = torch.FloatTensor([[2, 2], [2, 2]])
    reward = torch.FloatTensor([0, 0])
    value_pred = torch.zeros(2)
    action = torch.FloatTensor([0, 0])
    action_log_probs = torch.randn(2)
    dones = torch.FloatTensor([0, 0])

    # first step:
    state_t, action, reward, value_pred, action_log_probs, state_tp1, dones
    traj.append(state_t, action, reward, value_pred, action_log_probs, state_tp1, dones)

    # second step:
    state_t = state_tp1
    state_tp1 = torch.FloatTensor([[3, 3], [3, 3]])
    reward = torch.FloatTensor([5, 0])
    dones = torch.FloatTensor([1, 0]) # the first agent ends the trajectory.
    traj.append(state_t, action, reward, value_pred, action_log_probs, state_tp1, dones)

    # third step
    state_t = state_tp1
    state_tp1 = torch.FloatTensor([[4, 4], [4, 4]])
    reward = torch.FloatTensor([5, 5]) # dummy 5 that should not be taken into account when calculating returns
    dones = torch.FloatTensor([1, 1]) # both agents are done
    traj.append(state_t, action, reward, value_pred, action_log_probs, state_tp1, dones)

    # torchify the trajectories
    traj.torchify()
    assert traj.actions.size() == (3, 2, 1)
    assert traj.states.size() == (4, 2, 2)
    assert np.all(traj.states[0] == 1) # state at time 0
    assert np.all(traj.states[1] == 2) # state at time 1
    assert np.all(traj.states[2] == 3) # state at time 2
    assert np.all(traj.states[3] == 4) # state at tiem 3

    assert np.all(traj.dones[0] == 0) # does not end at time 1
    assert np.all(traj.dones[1] == torch.FloatTensor([[1], [0]])) # only one ends
    assert np.all(traj.dones[2] == torch.FloatTensor([[1], [1]])) # both ends

    assert np.all(traj.masks[0] == 1) # no masking
    assert np.all(traj.masks[1] == torch.FloatTensor([[1], [1]])) # no masking
    assert np.all(traj.masks[2] == torch.FloatTensor([[0], [1]])) # only mask first

    returns = calculate_returns(traj.rewards, 0.99, traj.masks)

    # the 5 from step 3 should be masked for the first agent but not the second agent
    assert np.all(returns[0] == torch.FloatTensor([[5*0.99], [5*0.99*0.99]]))
