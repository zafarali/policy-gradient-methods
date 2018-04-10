# `pg_methods` organization

This document describes how `pg_methods` is organized:

## `pg_methods.algorithms`

Contains implementations of common algorithms. Right now the following are implemented:

1. [`VanillaPolicyGradient`](./algorithms/REINFORCE/algorithm.py) contains the implementation of REINFORCE vanilla policy gradient. Baselines are optional and supported.


## [`pg_methods.data`](./data/)

Contains various utilities to handle data collection and storage from environments. This should be the future home of experience replay and things like that.

1. [`obtain_trajectories`](./data/collectors.py) Conducts a rollout in the environment
2. [`MultiTrajectory`](./data/storage.py) Stores rollouts from the environment. Has a `.torchify()` method to quickly convert the internal things to be used with PyTorch

## [`pg_methods.interfaces`](./interfaces/)

This contains some interfaces to go between PyTorch and OpenAI Gym.

Gym has a few Data objects: `Box`, `Discrete` etc. There are some utilities to automatically convert
between these types and PyTorch tensors. They contain functions like `gym2pytorch` and `pytorch2gym` that allows it to work
with the `PyTorchWrap` object.

1. [`ContinuousProcessor`](./interfaces/box_interfaces.py): Converts between `Box` datatype and PyTorch
2. [`SimpleDiscreteProcessor`](./interfaces/discrete_interfaces.py): Converts the a sample from `Discrete` into a one hot vector.
3. [`OneHotProcessor`](./interfaces/discrete_interfaces.py): Converts the a sample from `Discrete` into a float that can be fed into PyTorch..


There are some wrappers and parallelized Gym interfaces:
1. [`PyTorchWrap`](./interfaces/wrappers.py) Interface between a single gym instance and pytorch
2. [`make_parallelized_gym_env`](./interfaces/parallelized_gym.py) Interface between multiple gym environments running in parallel in pytorch. 


## [`pg_methods.networks`](./networks/)

This contains some common neural networks often used as function approximators for policies. Examples are:

1. [`MLP_factory`](./networks/approximators.py): creates a simple MLP
2. [`MLP_factory_two_heads`](./networks/approximators.py): Used to create networks with a shared body and two heads with different parameters.
3. [`SharedActorCritic`](./networks/actor_critic.py) -- (WIP) used for creating actor critic algorithms with shared heads and bodies.

## [`pg_methods.objectives`](./objectives/)

Contains `PolicyGradientObjective` which actually should be the REINFORCE objective. 
(Maybe we should consider changing this in a future release?), and `NaturalPolicyGradientObjective` which is not yet implemented.

## [`pg_methods.baselines.py`](./baselines.py)

Right now contains two baseline functions: `MovingAverageBaseline`, `FunctionApproximatorBaseline`. 

## [`pg_methods.gradients.py`](./gradients.py)

Functions to help calculate gradients for the policy gradient objectives. These are all found in `PolicyGradientObjective`, but a few things that are useful to play with are:

1. `calculate_returns(rewards, discount, masks)` calculates retuns given rewards, discount factor and masks. The arguments are usually found by using `MultiTrajectory` 
2. `calculate_policy_gradient_terms(log_probs, advantage)` calculates policy gradient terms `logprob * advantage` (no mean happens here.)

## [`pg_methods.policies.py`](./policies.py)

Includes common policies used in reinforcement learning.

All policies take in a function approximator as the first argument. This is a torch module like a neural network.

### `RandomPolicy`

Categorical agent that acts randomly.

### `CategoricalPolicy`

### `BernoulliPolicy`

### `GaussianPolicy`

Actions are picked according to a Gaussian distribution parameterized by `mu` and `sigma`. 
Note that in this case the function approximator should return two outputs corresponding to the `mu` and `sigma`


## [`pg_methods.utils`](./utils/)

1. [`pg_methods.utils.experiment`](./utils/experiment.py): Contains some tools to handle experiments and setup policies quickly
2. [`pg_methods.utils.logger`](./utils/logger.py): Should contain things to log data. Will be the future home of the Tensorboard logger etc.
2. [`pg_methods.utils.plotting`](./utils/plotting.py): Tools for plotting results of a run.
