"""
Implementation of the REINFORCE algorithm as found in Suttons book.
"""
import sys
import numpy as np
from pg_methods.algorithms.common import Algorithm
from pg_methods.utils.data import obtain_trajectory
import pg_methods.utils.gradients as gradients
from torch.nn.utils import clip_grad_norm 

class REINFORCE(Algorithm):
    def __init__(self, 
                 environment,
                 policy,
                 policy_optimizer,
                 state_processor,
                 action_processor,
                 gamma=0.99,
                 baseline=None,
                 logger=None,
                 max_horizon=None,
                 use_cuda=False):
        super().__init__(environment, policy, logger, use_cuda)

        self.max_horizon = max_horizon if max_horizon is not None else sys.maxsize
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.gamma = gamma

    def run(self, n_episodes, verbose=False):

        rewards = []
        losses = []
        for i in range(n_episodes):
            trajectory = obtain_trajectory(self.environment,
                                           self.policy,
                                           self.state_processor,
                                           self.action_processor,
                                           baseline_function=self.baseline)

            advantages = gradients.calculate_advantages(trajectory.rewards, self.gamma, trajectory.baselines)

            if self.baseline is not None:
                self.baseline.update_baseline(advantages.data.tolist())
    
            loss, loss_terms = gradients.calculate_loss_terms(trajectory.log_probs, advantages)
    
            if i % 100 == 0 and verbose:
                print('episode {}/{}: loss {} episode_reward {}'.format(i, n_episodes, loss.data[0], sum(trajectory.rewards)))
                # print('baseline values: {}'.format(trajectory.baselines))
                # print('step rewards: {}'.format(trajectory.rewards))
                # print('advantages: {}'.format(advantages.data.tolist()))

            for loss_term in loss_terms:
                self.policy_optimizer.zero_grad()
                loss_term.backward()
                clip_grad_norm(self.policy.fn_approximator.parameters(), 40)
                self.policy_optimizer.step()
        
            rewards.append(sum(trajectory.rewards))
            losses.append(loss.data[0])
            self.log(episode=i, reward=sum(trajectory.rewards))

        return np.array(rewards), losses

