import logging

import gym

from pg_methods.interfaces.discrete_interfaces import OneHotProcessor
from pg_methods.interfaces.box_interfaces import ContinuousProcessor

def infer_space_information(space):
    if isinstance(space, gym.spaces.Discrete):
        return {'type':'discrete', 'possible_values': space.n}
    elif isinstance(space, gym.spaces.Box):
        return {'type': 'continuous', 'shape': space.shape }


class PyTorchWrapper(gym.Wrapper):
    """
    This is a wrapper for gym environments so that they can be used
    easily with PyTorch neural networks. It makes code a bit cleaner
    and allows the same algorithm to interface with both 
    parallelized_gym.VecEnv and regular gym
    """
    _pytorch = True
    def __init__(self, env, observation_processor=None, action_processor=None, use_cuda=False):
        """
        A PyTorch wrapper for a gym environment.
        If any of the observation_processor or action_processor is None, then we will try to
        infer it.
        :param env: The environment to wrap
        :param observation_processor: The observation processor to use
        :param action_processor: The action processor to use
        :param use_cuda: use cuda when producing the Variable.
        """

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.observation_space_info = infer_space_information(self.env.observation_space)
        self.action_space_info = infer_space_information(self.env.action_space)

        if observation_processor is None:
            if self.observation_space_info['type'] == 'discrete':
                logging.warning('Defaulting to use OneHotProcessor for Discrete Observation Space')
                self.observation_processor = OneHotProcessor(self.observation_space_info['possible_values'], action_processor=False)
            elif self.observation_space_info['type'] == 'continuous':
                logging.warning('Defaulting to use ContinuousProcessor for Box Observation Space')
                self.observation_processor = ContinuousProcessor()
            else:
                raise TypeError('Unknown observation type, you must specify a observation processor for this.')
        else:
            self.observation_processor = observation_processor

        if action_processor is None:
            if self.action_space_info['type'] == 'discrete':
                logging.warning('Defaulting to use OneHotProcessor for Discrete Action Space')
                self.action_processor = OneHotProcessor(self.action_space_info['possible_values'], action_processor=True)
            elif self.action_space_info['type'] == 'continuous':
                logging.warning('Defaulting to use ContinuousProcessor for Box Action Space')
                self.action_processor = ContinuousProcessor()
            else:
                raise TypeError('Unknown observation type, you must specify a observation processor for this.')
        else:
            self.action_processor = action_processor

        self.use_cuda = use_cuda

    def variable_wrap(self, tensor):
        if self.use_cuda:
            return tensor.cuda()
        else:
            return tensor

    def _step(self, action):
        print('Input action:', action)
        action = self.convert_pytorch_action_to_gym(action)
        print('Converted to gym action:', action)
        state, reward, done, info = self.env.step(action)
        return (self.convert_gym_state_to_pytorch(state), reward, done, info)

    def reset(self, **kwargs):
        return self.convert_gym_state_to_pytorch(self.env.reset(**kwargs))

    def _render(self, mode='human'):
        self.env.render(mode)

    def _close(self):
        return self.env.close()

    def _seed(self, seed):
        return self.env.seed(seed)

    def convert_pytorch_action_to_gym(self, action):
        """
        Note this only works for simple action
        """
        return self.action_processor.pytorch2gym(action)

    def convert_gym_state_to_pytorch(self, state):
        state = self.observation_processor.gym2pytorch(state)
        return self.variable_wrap(state)
