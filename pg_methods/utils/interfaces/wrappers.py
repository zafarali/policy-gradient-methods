import gym
import numpy as np
import torch
from torch.autograd import Variable

def infer_space_information(space):
    if isinstance(space, gym.spaces.Discrete):
        return {'type':'discrete', 'possible_values': space.n}
    elif isinstance(space, gym.spaces.Box):
        return {'type': 'continuous', 'shape': space.shape }

def number_convert(number):
    if type(number) in [np.int64, np.int32]:
        return int(number)
    if type(number) in [np.float32, np.float64]:
        return float(number)
    else:
        return number

class PyTorchWrapper(gym.Wrapper):
    """
    This is a wrapper for gym environments so that they can be used
    easily with PyTorch neural networks. It makes code a bit cleaner
    and allows the same algorithm to interface with both 
    parallelized_gym.VecEnv and regular gym
    """
    _pytorch = True
    def __init__(self, env, onehot=False, use_cuda=False):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.observation_space_info = infer_space_information(self.env.observation_space)
        self.action_space_info = infer_space_information(self.env.action_space)
        if onehot and self.observation_space_info['type'] == 'discrete':
            self.observation_onehot = onehot
        elif onehot:
            raise KeyError('Only discrete spaces can have onehot action spaces')
        else:
            self.observation_onehot = False
        self.use_cuda = use_cuda

    def variable_wrap(self, tensor):
        if self.use_cuda:
            return Variable(tensor).cuda()
        else:
            return Variable(tensor)

    def _step(self, action):
        action = self.convert_pytorch_action_to_gym(action)
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
        if isinstance(action, Variable):
            action = action.data

        # check sizes and convert. 
        # TODO: this only works with single actions (i.e.)
        assert tuple(action.size()) == (1, ) or tuple(action.size()) == (1, 1)
        if len(action.size()) == 1:
            action = action.view(1, 1)
        
        if self.action_space_info['type'] == 'continuous':
            return list(map(float, action.tolist()[0]))
        elif self.action_space_info['type'] == 'discrete':
            return int(action.tolist()[0][0])

    def convert_gym_state_to_pytorch(self, state):
        if type(state) not in [list, np.array]:
            state = [number_convert(state)]
        if self.observation_onehot:
            one_hot = torch.zeros(len(state), self.observation_space_info['possible_values'])
            for i, val in enumerate(state):
                one_hot[i, val] = 1
            one_hot = torch.FloatTensor(one_hot)
            return self.variable_wrap(one_hot)
        elif self.observation_space_info['type'] == 'continuous':
            if type(state) is np.ndarray:
                state = torch.from_numpy(state).view(1, -1)
            else:
                state = torch.FloatTensor(state).view(1, -1)
            return self.variable_wrap(state)
        else:
            return self.variable_wrap(torch.FloatTensor(state).view(1, -1))
