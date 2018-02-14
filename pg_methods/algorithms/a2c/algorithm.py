"""
Implementation of Synchronous advantage actor critic. 
"""
from ..common import Algorithm
from ...utils.objectives import PolicyGradientObjective
from ...utils.data import obtain_trajectories

class A2C(Algorithm):
    def __init__(self,
                 environment,
                 actor_critic_architecture,
                 gamma=0.99,
                 objective=PolicyGradientObjective(),
                 baseline=None,
                 logger=None,
                 max_horizon=None,
                 time_mean=False,
                 use_cuda=False):
        super().__init__()
        pass

    def run(self, n_episodes, steps_before_update, verbose=False):
        for global_t in range(n_episodes*steps_before_update):
            partial_trajectory = obtain_trajectories(policy, steps_before_update,reset=False,value_function=critic,verbose=verbose)

