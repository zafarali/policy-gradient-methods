"""
Implements some common baselines
"""
import torch
from torch.autograd import Variable
import numpy as np
import logging
import copy

class Baseline(object):
    optimizer = None
    def update_baseline(self, trajectory, returns=None):
        """
        Update the baseline
        :param trajectory: A trajectory object
        :param returns: The calculated return G_{t+1}
        :return:
        """
        raise NotImplementedError('need to implement!')

    def __call__(self, state):
        return 0


class MovingAverageBaseline(Baseline):
    def __init__(self, beta):
        super().__init__()
        self.value = 0
        self.beta = beta
        self.initiated = False

    def update_baseline(self, trajectory, returns=None):
        if not self.initiated:
            self.value = returns.numpy().mean()
            self.initiated = True
        else:
            self.value = self.value * (1-self.beta) + self.beta * returns.numpy().mean()

        return 0

    def __call__(self, state):
        # first dim is the number o
        return self.value * torch.ones(state.size()[0])

    def __str__(self):
        return 'MovingAverageBaseline({})'.format(self.beta)


class FunctionApproximatorBaseline(Baseline):
    def __init__(self, fn_approximator, optimizer=None):
        """
        A function approximator baseline that estimates
            V(s)= Expected future return

        :param fn_approximator: the function approximator to use
        :param optimizer: the torch.optim.Optimizer instance.
                          If optimizer is None no update step will be run
                          And it will be upto the policy optimizer to do it
                          This will be useful when using shared architectures (yet untested)
        """
        super().__init__()
        self.fn_approximator = fn_approximator
        self.optimizer = optimizer

    def update_baseline(self, trajectory, returns, tro=False):
        """
        Function to update the baseline function approximator

        :param trajectory:
        :param returns:
        :tro: Use Trust Region Optimization for value function update (ref: https://arxiv.org/pdf/1506.02438.pdf)
        """
        # handle MultiTrajectory vs Trajectory
        # for backwards compatability, # TODO: might want to remove this in the future
        if len(trajectory.states.size()) == 3:
            traj_len, n_envs, state_size = trajectory.states.size()
        else:
            traj_len, state_size = trajectory.states.size()
            n_envs = 1

        # reshape so that it's batch x state_space
        current_states = trajectory.states[:-1].view( (traj_len-1 )* n_envs, state_size)

        # get the value prediction for these states
        value_estimates = self.fn_approximator(current_states)
        if not isinstance(returns, Variable): returns = Variable(returns)

        # take the difference between the returns actually observed
        # reshape so that it is batch x returns
        returns = returns.view((traj_len-1)*n_envs, 1)
        loss = (returns - value_estimates)**2

        # apply masks for terminated trajectories
        loss = torch.mean(loss*Variable(trajectory.masks.view_as(loss).float()))

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()

            # TRO Constraint

            if tro:
                sigma_2 = loss/traj_len
                model_new = copy.deepcopy(self.fn_approximator)
                optim = torch.optim.RMSprop(model_new.parameters(), lr=1e-5)
                optim.step()

                value_estimates_new = model_new(current_states)

                diff = (value_estimates_new - value_estimates)**2
                diff = torch.mean(diff*Variable(trajectory.masks.view_as(diff).float()))
                diff = diff/(2*sigma_2)

                if diff.data.numpy()[0] < 1e-10: # TODO: See if any alternative to hardcoding epsilon
                    self.optimizer.step()
            else:
                self.optimizer.step()

        return loss

    def __call__(self, state):
        value_estimate = self.fn_approximator(state).data
        return value_estimate
    def __str__(self):
        return 'FunctionApproximatorBaseline()'


class NeuralNetworkBaseline(FunctionApproximatorBaseline):
    def __init__(self, fn_approximator, optimizer, bootstrap=None, bootstrap_gamma=None):
        logging.warning('NeuralNetworkBaseline is now a special case of `FunctionApproximatorBaseline`. '
                        'Please use that going forward to keep having the latest updates!')
        super().__init__(fn_approximator, optimizer)
