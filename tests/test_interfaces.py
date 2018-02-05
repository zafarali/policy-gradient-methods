import gym
import torch
from torch.autograd import Variable
from pg_methods.utils.interfaces.parallelized_gym import make_parallelized_gym_env 
from pg_methods.utils.interfaces.wrappers import PyTorchWrapper
import pg_methods.utils.policies as policies
from pg_methods.utils.networks import MLP_factory

"""
These set of tests will check if the interfaces between 
policies.Policy and gym is seamless.
"""
def get_gaussian_action(state, *args):
    policy = policies.GaussianPolicy(MLP_factory(state.size()[1], output_size=2))
    action = policy(state)
    return action

def get_multinomial_action(state, action_space_size):
    if len(state.size()) != 2:
        policy = policies.MultinomialPolicy(MLP_factory(1, output_size=action_space_size))
    else:
        policy = policies.MultinomialPolicy(MLP_factory(state.size()[1], output_size=action_space_size))
    action = policy(state)
    return action

def get_bernoulli_action(state, *args):
    policy = policies.BernoulliPolicy(MLP_factory(state.size()[1], output_size=1))
    action = policy(state)
    return action

def check_state_details(state):
    print(state)
    assert type(state) is Variable
    assert len(state.size()) == 2
    assert state.size()[0] == 1
    return True

def test_single_env_continous_state_discrete_actions():
    env = gym.make('CartPole-v0')
    env = PyTorchWrapper(env)
    state = env.reset()
    assert check_state_details(state)
    action, log_probs = get_multinomial_action(state, env.action_space.n)
    assert check_state_details(env.step(action)[0])

def test_single_env_discrete_state_discrete_actions():
    env = gym.make('FrozenLake-v0')
    env = PyTorchWrapper(env, onehot=False)
    state = env.reset()
    assert check_state_details(state)
    action, log_probs = get_multinomial_action(state, env.action_space.n)
    assert check_state_details(env.step(action)[0])

def test_single_env_discrete_state_one_hot_discrete_actions():
    env = gym.make('FrozenLake-v0')
    env = PyTorchWrapper(env, onehot=True)
    state = env.reset()
    assert check_state_details(state)
    assert state.size()[1] == env.observation_space.n
    assert state[0, 0].data[0] == 1
    assert state[0, 1:].sum().data[0] == 0
    action, log_probs = get_multinomial_action(state, env.action_space.n)
    assert check_state_details(env.step(action)[0])

def test_single_env_continous_state_continous_actions():
    env = gym.make('MountainCarContinuous-v0')
    env = PyTorchWrapper(env)
    state = env.reset()
    assert check_state_details(state)
    action, log_probs = get_gaussian_action(state)
    assert check_state_details(env.step(action)[0])

def test_multi_env_continous_state_discrete_actions():
    env = make_parallelized_gym_env('CartPole-v0', 0, 2)
    state = env.reset()
    assert tuple(state.size()) == (2, 4)
    assert type(state) is Variable
    action, log_probs = get_multinomial_action(state, env.action_space.n)
    state, _, _, _ = env.step(action)
    assert type(state) is Variable
    assert tuple(state.size()) == (2, 4)

# def test_multi_env_discrete_state_discrete_actions():
#     env = make_parallelized_gym_env('FrozenLake-v0',0, 2)

# def test_multi_env_discrete_state_one_hot_discrete_actions():
#     env = make_parallelized_gym_env('FrozenLake-v0', 0, 2)

def test_multi_env_continous_state_continous_actions():
    env = make_parallelized_gym_env('MountainCarContinuous-v0', 0, 2)
    state = env.reset()
    action, log_probs = get_gaussian_action(state, env.action_space.shape)
    assert type(state) is Variable
    assert tuple(state.size()) == (2, 2)
    state, _, _, _ = env.step(action)
    assert type(state) is Variable
    assert tuple(state.size()) == (2, 2)
