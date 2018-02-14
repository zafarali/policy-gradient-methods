"""
Contains functions and objects for data collection
"""
import torch
import sys
import logging
from torch.autograd import Variable
from .storage import Transition, Trajectory, MultiTrajectory

def obtain_trajectories_single_env(environment,
                                   policy,
                                   steps,
                                   reset=True,
                                   value_function=None,
                                   verbose=False):
    """
    Obtain a single trajectory by executing a policy in an environment
    :param environment: a single instance of an environment
    :param policy: a policy that returns a single action
    :param state_processor: 
    :param action_processor:
    :param baseline_function: the baseline function to use
    :param max_steps: the maximum number of steps
    :param verbose: print to stdout
    """

    steps = sys.maxsize if steps is None else steps
    trajectory = Trajectory()

    if not reset:
        logging.warn('Reset is always called in a single environment')
    state_t = environment.reset()

    if verbose: print("Start state: {}".format(state_t))

    for t in range(steps):
        action, log_prob = policy(state_t)

        if verbose: print('Action taken: {}'.format(action))
        
        state_tp1, reward, done, info = environment.step(action)
        
        if verbose: print('State: {}, reward: {}, done {}'.format(state_tp1, reward, done))
        
        baseline = value_function(state_t) if value_function is not None else 0
        trajectory.append(state_t, action, reward, baseline, log_prob, state_tp1, done)

        state_t = state_tp1

        if done:
            break

    return trajectory


def obtain_trajectories_parallel_env(environment,
                                     policy,
                                     steps,
                                     reset=True,
                                     value_function=None,
                                     verbose=False):
    """
    
    """
    assert environment._parallel, 'Your environment must allow for parallel execution'
    assert environment._vectorized , 'Your environment must alllow for the execution of multiple actions simultaneously'

    trajectory = MultiTrajectory(environment.n_envs)

    state_t = environment.reset() if reset else environment.state

    if verbose: print("Start state: {}".format(state_t))


    for t in range(steps):
        action, log_prob = policy(state_t)

        if verbose: print('Action taken: {}'.format(action))
        
        state_tp1, reward, done, info = environment.step(action)
        if verbose: print('State: {}, reward: {}, done {}, info {}'.format(state_tp1.data, reward, done, info))
        
        value_estimate = value_function(state_t) if value_function is not None else torch.FloatTensor([[0]*environment.n_envs])
        trajectory.append(state_t, action, reward, value_estimate, log_prob, state_tp1, info['trajectory_done'])

        state_t = state_tp1

        if not info['environments_left']:
            # print('All environments done. Closing.')
            break

    return trajectory


def obtain_trajectories(environment,
                        policy,
                        steps,
                        reset=True,
                        value_function=None,
                        verbose=False):
    
    assert environment._pytorch, 'Your environment must be PyTorch compatible'
    if hasattr(environment, '_parallel'):
        if verbose: logging.info('Parallel Environment Detected')
        return obtain_trajectories_parallel_env(environment, policy, steps, reset, value_function, verbose)
    elif hasattr(environment, '_vectorized'):
        if verbose: logging.info('Vectorized Environment Detected')
        raise NotImplementedError('No support for vanilla vectorized environments yet')
    else:
        if verbose: logging.info('Other Environment Detected')
        return obtain_trajectories_single_env(environment, policy, steps, reset, value_function, verbose)
