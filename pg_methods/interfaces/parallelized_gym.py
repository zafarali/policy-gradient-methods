import gym
import numpy as np
try:
    import roboschool
except ImportError as e:
    print('Roboschool not installed. ')
import torch
from torch.autograd import Variable
from torch.multiprocessing import Process, Pipe
import logging
from .wrappers import infer_space_information
from .discrete_interfaces import OneHotProcessor, SimpleDiscreteProcessor
from .box_interfaces import ContinuousProcessor
from .common_interfaces import NUMBERS, number_convert
"""
Code to parallelize gym environments obtained from
https://github.com/openai/baselines/blob/b5be53dc928bc19c39bce2a3f8a4e7dd0374f1dd/baselines/common/vec_env/subproc_vec_env.py
It has been modified to deal with PyTorch Tensors and Variables.
"""

class VecEnv(object):
    """
    Vectorized environment base class
    """
    _vectorized = True
    _pytorch = True
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        raise NotImplementedError
    def reset(self):
        """
        Reset all environments
        """
        raise NotImplementedError
    def close(self):
        pass


def worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker that handles executes action in an environmen
    and returns the results back
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

"""
Multiprocessing environment runs many envs in parallel.
Deals directly with PyTorch tensors and Variables.
(!) Note that you can access both the dones for each transition as usual by the return 
or you can access if the trajectory had already experienced a done from infos['trajectory_done']
"""
class SubprocVecEnv(VecEnv):
    _parallel = True
    def __init__(self, env_fns,
                 mask_done_trajectories=False,
                 action_processor=None,
                 observation_processor=None,
                 use_cuda=False,
                 onehot=False):
        """

        :param env_fns:
        :param mask_done_trajectories:
        :param action_processor:
        :param observation_processor:
        :param use_cuda:
        :param onehot:
        """
        self.use_cuda = use_cuda
        self.closed = False
        nenvs = len(env_fns)
        self.n_envs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        self.mask_done_trajectories = mask_done_trajectories

        self.observation_space_info = infer_space_information(self.observation_space)
        self.action_space_info = infer_space_information(self.action_space)

        # note that if you edit code here, you should check interfaces.wrappers if PyTorchWrapper
        # needs any modifications.
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

        self.reset_dones()
        self.state = self.reset()

    def reset_dones(self):
        self.dones = [0] * self.n_envs

    def step(self, actions):
        """
        Execute a list of actions in each environment
        """
        if isinstance(actions, torch.autograd.Variable):
            actions = actions.data

        actions = actions.cpu()

        for remote, action in zip(self.remotes, actions.tolist()):
            # convert action to gym and execute it in the environment
            if type(action) in NUMBERS:
                action = [number_convert(action)]

            action = self.action_processor.pytorch2gym(action)
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]

        obs, rews, dones, infos = zip(*results)

        # convert the observations into pytorch tensors

        obs = [self.observation_processor.gym2pytorch(ob).view(-1) for ob in obs]
        # stack and wrap.
        obs = self.variable_wrap(torch.stack(obs))

        for n, done in enumerate(list(dones)):
            if done or self.dones[n]:
                self.dones[n] = 1


        self.state = obs
        modified_infos =  {
                           'subprocess':infos,
                           'environments_left': self.n_envs - sum(self.dones),
                           'trajectory_done': torch.ByteTensor(self.dones)
                          }
        return obs, torch.FloatTensor(rews), torch.ByteTensor(dones), modified_infos

    def reset(self):
        """
        Reset all the environments
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        self.reset_dones()
        # handle one hot encoding here
        self.state = self.variable_wrap(torch.stack([self.observation_processor.gym2pytorch(remote.recv()).view(-1) for remote in self.remotes]))
        return self.state

    def variable_wrap(self, tensor):
        if self.use_cuda:
            return Variable(tensor).cuda()
        else:
            return Variable(tensor)

    def reset_task(self):
        """
        Reset the task
        """
        for remote in self.remotes:
            remote.send(('reset_task', None))
        self.reset_dones()
        return self.variable_wrap(torch.stack([self.observation_processor.gym2pytorch(remote.recv()).view(-1) for remote in self.remotes]))

    def close(self):
        """
        Close all environments
        """
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)


def make_env(env_id, seed, rank, logger=None):
    logging.warn('Atari is not supported. See https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py for atari support')
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _thunk

def make_parallelized_gym_env(env_id, seed, n_workers, action_processor=None, observation_processor=None):
    env_fns = [make_env(env_id, seed, i) for i in range(n_workers)]
    return SubprocVecEnv(env_fns, action_processor=action_processor, observation_processor=observation_processor)
