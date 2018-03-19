"""
Implementation of the REINFORCE algorithm as found in Suttons book.
"""
import sys
import numpy as np
import torch
from pg_methods.algorithms.common import Algorithm
from ...data import obtain_trajectories
from ... import gradients
from ...objectives import PolicyGradientObjective
from torch.nn.utils import clip_grad_norm 
import logging

class VanillaPolicyGradient(Algorithm):
    def __init__(self,
                 environment,
                 policy,
                 policy_optimizer,
                 gamma=0.99,
                 objective=PolicyGradientObjective(),
                 baseline=None,
                 logger=None,
                 max_horizon=None,
                 time_mean=False,
                 use_cuda=False):
        """
        Implements the batch version of REINFORCE:
            1. Sample a few trajectories
            2. Sum the logprob * adv for each sample
            3. take gradient to update parameters

        This is seen in Lecture 4a from the deep RL bootcamp:
        https://drive.google.com/file/d/0BxXI_RttTZAhY216RTMtanBpUnc/view

        :param environment: the (parallel) environment to query
        :param policy: the policy to use
        :param policy_optimizer: the optimizer to use for the policy
        :param gamma: the discount factor
        :param baseline: the baseline to use
        :param logger: the logger to use
        :param max_horizon: the maximum length of a trajectory
        :param use_cuda: use GPU tensors from torch
        """
        super().__init__(environment, policy, objective, logger, use_cuda)
        self.max_horizon = max_horizon if max_horizon is not None else sys.maxsize
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.time_mean = time_mean

    def run(self, n_episodes, verbose=False):
        rewards = []
        losses = []
        for i in range(n_episodes):
            trajectories = obtain_trajectories(self.environment,
                                               self.policy,
                                               sys.maxsize,
                                               reset=True,
                                               value_function=self.baseline)

            trajectories.torchify()
            returns = gradients.calculate_returns(trajectories.rewards, self.gamma, trajectories.masks)
            advantages = returns - trajectories.values
            
            if self.baseline is not None:
                baseline_loss = self.baseline.update_baseline(trajectories, returns)

            loss = self.objective(advantages, trajectories)

            # add the baseline loss to the overall loss to get a joint loss
            # this allows for shared architectures between policy and baseline
            # this is only run if the function approximator doesn't have an
            # associated baseline with it.
            if self.baseline is not None and self.baseline.optimizer is None:
                loss += baseline_loss

            if self.use_cuda:
                loss = loss.cuda()

            self.policy_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(self.policy.fn_approximator.parameters(), 40)
            self.policy_optimizer.step()

            reward_summary = torch.sum(trajectories.rewards * trajectories.masks.float(), dim=0) 
            if i % 100 == 0 and verbose:
                print('Episode {}/{}: loss {:3g} episode_reward {:3g}, average_value: {:3g}'.format(i, n_episodes,
                                                                                                    loss.data[0],
                                                                                                    reward_summary.mean(),
                                                                                                    trajectories.values.mean()))
                print('Longest Trajectory {} / Individual rewards: {}'.format(trajectories.masks.sum(dim=0).max(),
                                                                              reward_summary.tolist()))
            rewards.append(torch.sum(trajectories.rewards, dim=0).mean())
            losses.append(loss.data[0])
            self.log(episode=i, returns=returns, reward=reward_summary.mean(), trajectory=trajectories)
        
        return np.array(rewards), losses


class REINFORCE(Algorithm):
    """
    Implements the REINFORCE algorithm as mentioned in 
    pg 272 of 2nd edition in Suttons book.
    """
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
                 lr_scheduler=None,
                 use_cuda=False):
        super().__init__(environment, policy, logger, use_cuda)
        logging.warning('Use `VanillaPolicyGradient` for latest code base')

        self.max_horizon = max_horizon if max_horizon is not None else sys.maxsize
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.gamma = gamma
        self.lr_scheduler = lr_scheduler

    def run(self, n_episodes, max_steps=None, verbose=False):

        rewards = []
        losses = []
        for i in range(n_episodes):
            trajectory = obtain_trajectories(self.environment,
                                            self.policy,
                                            steps=max_steps,
                                            value_function=self.baseline)

            trajectory.torchify()

            returns = gradients.calculate_returns(trajectory.rewards, self.gamma)
            advantages = returns - trajectory.values
            if self.baseline is not None:
                self.baseline.update_baseline(trajectory, advantages)
    
            policy_loss = gradients.calculate_policy_gradient_terms(trajectory.log_probs, advantages)
            policy_loss = policy_loss.sum(dim=0).mean()
            if i % 100 == 0 and verbose:
                probs = torch.exp(torch.stack(trajectory.log_probs))
                entropy = (-(probs * probs.log()).sum()).data[0]
                print('episode {}/{}: loss {} episode_reward {} policy entropy {:.2g}'.format(i, n_episodes, policy_loss.data[0], sum(trajectory.rewards)[0], entropy))
                # print('baseline values: {}'.format(trajectory.baselines))
                # print('step rewards: {}'.format(trajectory.rewards))
                # print('advantages: {}'.format(advantages.data.tolist()))

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm(self.policy.fn_approximator.parameters(), 40)
            self.policy_optimizer.step()

            if self.lr_scheduler is not None: self.lr_scheduler.step()
    
            rewards.append(sum(trajectory.rewards))
            losses.append(policy_loss.data[0])
            self.log(episode=i, reward=sum(trajectory.rewards), trajectory=trajectory)

        return np.array(rewards), losses

