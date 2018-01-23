import gym
import numpy as np
from torch.multiprocessing import Process, Pipe
import logging
"""
Code to parallelize gym environments obtained from
https://github.com/openai/baselines/blob/b5be53dc928bc19c39bce2a3f8a4e7dd0374f1dd/baselines/common/vec_env/subproc_vec_env.py
"""

class VecEnv(object):
    """
    Vectorized environment base class
    """
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


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        :param envs_fns: a list of functions that return gym environments to run in a subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
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


    def step(self, actions):
        """
        Execute a list of actions in each environment
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """
        Reset all the environments
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        """
        Reset the task
        """
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

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

def make_parallelized_gym_env(env_id, seed, n_workers):
    env_fns = [make_env(env_id, seed, i) for i in range(n_workers)]
    return SubprocVecEnv(env_fns)
