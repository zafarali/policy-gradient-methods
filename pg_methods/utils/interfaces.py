"""
A set of utilities to convert between pytorch and openai gym

States.gym2pytorch - converts a gym state to a pytorch tensor of size (1, -1)
Rewards.pytorch2gym - 
"""
import gym
import torch
from torch.autograd import Variable
import numpy as np

def gym2pytorch(tuple_):
    """
    Converts the gym tuple into a torch variable of size (1, -1)
    """
    return Variable(torch.from_numpy(np.array(tuple_).reshape(1, -1)))

def pytorch2array(tensor):
    """
    Converts a pytorch Tensor or Variable of size (1, -1)
    into a flat list 
    """
    if type(tensor) is Variable:
        return tensor.data.numpy().reshape(-1)
    elif type(tensor) is torch.Tensor:
        return tensor.numpy().reshape(-1)
    else:
        return tensor

def pytorch2list(tensor):
    return pytorch2array(tensor).tolist()

list2pytorch = gym2pytorch


## @TODO make this into a gym wrapper?
class SimpleStateProcessor(object):
    def __init__(self, environment_observation_space, one_hot=False, use_cuda=False, normalize=False):
        self.observation_space = environment_observation_space
        if isinstance(environment_observation_space, gym.spaces.Box):
            # continous environment
            self.continous = True
            self.state_size = environment_observation_space.shape
            if len(self.state_size) == 1:
                self.state_size = self.state_size[0]
            self.one_hot = False
            self.normalize = False
        else:
            self.continous = False
            self.one_hot = one_hot

            if self.one_hot:
                self.state_size = environment_observation_space.n
                self.normalize = False
                self.max_obs = environment_observation_space.n
            else:
                self.normalize = normalize
                self.state_size = 1
                self.max_obs = environment_observation_space.n
        self.use_cuda = use_cuda
    
    def state2pytorch(self, state_idx):

        if self.one_hot and not self.continous:
            state = np.zeros(self.state_size)
            state[self.state_idx] = 1
            state = Variable(torch.from_numpy(state.reshape(1, -1)))
            if self.use_cuda:
                return state.float().cuda()
            else:
                return state.float()
        else:
            state = None
            if not self.continous:
                state = Variable(torch.from_numpy(np.array([state_idx]).reshape(1, -1)))
            else:
                state = Variable(torch.from_numpy(np.array(state_idx).reshape(1, -1)))
                if self.normalize:
                    state = state / self.max_obs

            if self.use_cuda:
                return state.float().cuda()
            else:
                return state.float()
    def pytorch2state(self, tensor):
        if self.continous:
            return pytorch2list(tensor)
        else:
            list_state = list(map(int, pytorch2list(tensor)))
            if self.state_size == 1:
                return list_state[0]
            else:
                return list_state

class SimpleActionProcessor(object):
    def __init__(self, action_space, one_hot=False):
        self.action_space = action_space
        if action_space.n == 2:
            self.action_size = action_space.n - 1
        else:
            self.action_size= action_space.n

    def pytorch2gym(self, action):
        return int(pytorch2list(action)[0])

