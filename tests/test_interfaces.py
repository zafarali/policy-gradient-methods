import gym
import torch
from torch.autograd import Variable
from pg_methods.interfaces.parallelized_gym import make_parallelized_gym_env
from pg_methods.interfaces.wrappers import PyTorchWrapper
from pg_methods.interfaces.box_interfaces import ContinuousProcessor
from pg_methods.interfaces.discrete_interfaces import OneHotProcessor, SimpleDiscreteProcessor
import pg_methods.policies as policies
from pg_methods.networks import MLP_factory, MLP_factory_two_heads
import logging
"""
These set of tests will check if the interfaces between 
policies.Policy and gym is seamless.
"""
def get_gaussian_action(state, action_space_size, *args):
    policy = policies.GaussianPolicy(MLP_factory_two_heads(state.size()[1], hidden_sizes=[16], output_size=action_space_size))
    action = policy(state)
    return action

def get_multinomial_action(state, action_space_size):
    if len(state.size()) != 2:
        policy = policies.CategoricalPolicy(MLP_factory(1, output_size=action_space_size))
    else:
        policy = policies.CategoricalPolicy(MLP_factory(state.size()[1], output_size=action_space_size))
    action = policy(state)
    return action

def get_bernoulli_action(state, *args):
    policy = policies.BernoulliPolicy(MLP_factory(state.size()[1], output_size=1))
    action = policy(state)
    return action

def check_state_details(state):
    """
    Checks if a state is a:
    Variable
    has size 2
    has first dimension equal to 1
    :param state:
    :return:
    """
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
    env = PyTorchWrapper(env,
                         action_processor=SimpleDiscreteProcessor(),
                         observation_processor=SimpleDiscreteProcessor())
    state = env.reset()
    assert check_state_details(state)
    action, log_probs = get_multinomial_action(state, env.action_space.n)
    assert check_state_details(env.step(action)[0])

def test_single_env_discrete_state_one_hot_discrete_actions():
    env = gym.make('FrozenLake-v0')
    env = PyTorchWrapper(env) # by default this is a one hot processor
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
    action, log_probs = get_gaussian_action(state, env.action_space.shape[0])
    assert check_state_details(env.step(action)[0])

def test_multi_env_continous_state_discrete_actions():
    env = make_parallelized_gym_env('CartPole-v0', 0, 2, action_processor=SimpleDiscreteProcessor())
    state = env.reset()
    assert tuple(state.size()) == (2, 4)
    assert type(state) is Variable
    action, log_probs = get_multinomial_action(state, env.action_space.n)
    state, _, _, _ = env.step(action)
    assert type(state) is Variable
    assert tuple(state.size()) == (2, 4)

def test_multi_env_continous_state_continous_actions():
    env = make_parallelized_gym_env('MountainCarContinuous-v0', 0, 2)
    state = env.reset()
    action, log_probs = get_gaussian_action(state, env.action_space.shape[0])
    assert type(state) is Variable
    assert tuple(state.size()) == (2, 2)
    state, _, _, _ = env.step(action)
    assert type(state) is Variable
    assert tuple(state.size()) == (2, 2)

def test_multi_env_one_hot_state_one_hot_actions():
    env = make_parallelized_gym_env('FrozenLake-v0',0, 2)
    state = env.reset()
    assert state.size() == (2, env.observation_space.n)
    assert type(state) is Variable
    actions, log_probs = get_multinomial_action(state, env.action_space.n)
    assert actions.size() == (2,)
    state, _, _, _ = env.step(actions)
    assert type(state) is Variable
    assert state.size() == (2, env.observation_space.n)

def test_multi_env_continous_states_multiple_continous_actions():
    #TODO: change this to use something other than roboschool?
    try:
        import roboschool
        env = make_parallelized_gym_env('RoboschoolHumanoid-v1', 0, 2)
        state = env.reset()
        action, log_probs = get_gaussian_action(state, env.action_space.shape[0])
        assert type(state) is Variable
        assert tuple(state.size()) == (2, env.observation_space.shape[0])
        state, _, _, _ = env.step(action)
        assert type(state) is Variable
        assert tuple(state.size()) == (2, env.observation_space.shape[0])
    except ImportError:
        logging.warning('No roboschool was installed therefore `test_multi_env_continous_states_multiple_continous_actions` could not be run')
        pass
