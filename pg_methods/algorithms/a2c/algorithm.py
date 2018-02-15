"""
Implementation of Synchronous advantage actor critic. 
"""
from ..common import Algorithm
from ...utils.objectives import PolicyGradientObjective
from ...utils.data import obtain_trajectories
from .ac_utils import ActorCriticController

class A2C(Algorithm):
    def __init__(self,
                 environment,
                 actor_critic,
                 gamma=0.99,
                 objective=PolicyGradientObjective(),
                 logger=None,
                 time_mean=False,
                 use_cuda=False):
        super().__init__(environment, actor_critic, objective, logger, use_cuda)
        self.gamma = gamma
        self.time_mean = time_mean
        self.ac_controller = ActorCriticController(actor_critic)

    def run(self, n_episodes, steps_before_update, verbose=False):
        episode_counter = 0
        for global_t in range(n_episodes * steps_before_update):
            # obtain a partial trajectory that is of length steps_before_update
            # or smaller depending on how long the episodes last

            if self.environment.all_done:
                episode_counter += self.environment.num_envs
                if episode_counter >= n_episodes:
                    break

            # obtain a partial trajectory
            partial_trajectory = obtain_trajectories(self.environment,
                                                     self.ac_controller.policy,
                                                     steps_before_update,
                                                     reset=self.environment.all_done,
                                                     value_function=self.ac_controller.critic,
                                                     verbose=verbose)
            pass
