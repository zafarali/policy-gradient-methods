# Policy Gradient Methods

PyTorch implementation of policy gradient methods.

*NOTE* This repository is still work in progress! As I continue to try to break things down into modular and reusable parts things might break. However, I will try to ensure the cases in tests keep passing.

## Installation

This library only works with Python 3.5+. If you are using Python 2.7 you should upgrade immediately. 
The requirements for this library can be found in `requirements.txt`. To install this library you can use pip:

```bash
pip install -e .
```

The `-e` indicates that the library will be installed in development mode. You can then check if it works by opening up python and typing:

```python
import pg_methods
print(pg_methods.__version__) # should print 0
```

## Tests

There are tests for components in this library under `./tests/`. You can run them by executing `python -m pytest ./tests --verbose`.

## Philosophy

There are a few [good](https://github.com/vitchyr/rlkit) [reinforcement learning](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) [reinforcement](https://github.com/ikostrikov/pytorch-ddpg-naf) [algorithm](https://github.com/jingweiz/pytorch-rl) algorithm implementations in pytorch. 
There are many ones in [Tensorflow](https://github.com/openai/baselines), [Theano](https://github.com/rll/rllab) and [Keras](https://github.com/keras-rl/keras-rl).
The main thing lacking in the PyTorch implementations is extensibility/modularity. Sure I would love to run this one algorithm on all environments ever. But sometimes it's just the little parts that are useful. For example,
a good utility to calculate [discounted future returns with masks](https://github.com/zafarali/policy-gradient-methods/blob/f7fc0b2ba06c208f97a73f1b9f0bdd0507fd2f54/pg_methods/gradients.py#L5-L24).
Or the [REINFORCE objective](https://github.com/zafarali/policy-gradient-methods/blob/f7fc0b2ba06c208f97a73f1b9f0bdd0507fd2f54/pg_methods/objectives/objectives.py#L40-L74) itself. 
Maybe you want to try a new kind of [baseline](https://github.com/zafarali/policy-gradient-methods/blob/master/pg_methods/baselines.py)? The goal of this library is to allow you to do all of these things. 
Sort of like LEGO. Arguably, more important than having a long script with the algorithm, is having the components to make new ones. This is one thing I find frustrating with baselines, all the algorithms are in their own folders, with only marginal code sharing. 
I've already used some stuff from here in [some](https://github.com/zafarali/deep-subsets/blob/master/experiments/integer_version_set2subset_RL.py) (old version of `pg_methods`) of [my projects](https://github.com/zafarali/better-sampling/blob/master/rvi_sampling/samplers/RVISampler.py) (soon to be released).

To see how the code is organized see [./pg_methods/README.md](./pg_methods/README.md)

## Algorithms Implemented

1. Vanilla Policy Gradient (`pg_methods.algorithms.VanillaPolicyGradient`)

### To be implemented (contributions welcome!)

- [ ] Synchronous Advantage Actor Critic
- [ ] Asynchronous Advantage Actor Critic
- [ ] Natural Policy Gradient
- [ ] Trust Region Policy Optimization
- [ ] Proximal Policy Optimization

etc.

## other opportunities to contribute

See [projects](https://github.com/zafarali/policy-gradient-methods/projects). Things like new `objectives`, `baselines` `optimizers`, `replay_memory`s are all good contributions!

Also what would be cool is a large scale benchmarking script so that we can run all the algorithms to see how they perform on different gym environments.

### Some performance graphs (soon to improve)

![rewards-plots](https://user-images.githubusercontent.com/6295292/37562262-a87510b0-2a3a-11e8-95bf-e3b8799bd546.png)

I'm working to get `roboschool` installed on the ComputeCanada clusters so i can run for longer. To install roboschool on your local machine you can [try this script](https://gist.github.com/zafarali/7186f8790c8c4288a9c4e72c04f8c8ce)

![rewards-roboschool](https://user-images.githubusercontent.com/6295292/37562263-aa016ea6-2a3a-11e8-95cc-beb68a4caa42.png)

## Example

Here is an example script of how to get started with the `VanillaPolicyGradient` algorithm. We expect other algorithms to have similar interfaces.

```python
from pg_methods import interfaces
from pg_methods.algorithms.REINFORCE import VanillaPolicyGradient
from pg_methods.baselines import FunctionApproximatorBaseline
from pg_methods.utils import experiment

env = interfaces.make_parallelized_gym_env('CartPole-v0', seed=4, n_workers=2)
experiment_logger = experiment.Experiment({'algorithm_name': 'VPG'}, './')
experiment_logger.start()
fn_approximator, policy = experiment.setup_policy(env, hidden_non_linearity=nn.ReLU, hidden_sizes=[16, 16])
optimizer = torch.optim.SGD(fn_approximator.parameters(), lr=0.01)

# setting up a baseline function
baseline_approximator = MLP_factory(env.observation_space_info['shape'][0],
                               [16, 16],
                               output_size=1,
                               hidden_non_linearity=nn.ReLU)
baseline_optimizer = torch.optim.SGD(baseline_approximator.parameters(), lr=0.01)
baseline = FunctionApproximatorBaseline(baseline_approximator, baseline_optimizer)

algorithm = VanillaPolicyGradient(env, policy, optimizer, gamma=0.99, baseline=baseline)

rewards, losses = algorithm.run(1000, verbose=True)

experiment_logger.log_data('rewards', rewards.tolist())
experiment_logger.save()
```    

More example scripts can be seen in [./experiments/](./experiments)