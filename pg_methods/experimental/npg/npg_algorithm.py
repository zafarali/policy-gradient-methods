import copy
import numpy as np
import sys
import time
from pg_methods import interfaces
from pg_methods.interfaces import parallelized_gym
from pg_methods.utils import experiment
from pg_methods.conjugate_gradient import conjugate_gradient_algorithm
from pg_methods.baselines import MovingAverageBaseline, FunctionApproximatorBaseline
from pg_methods.data import obtain_trajectories
from pg_methods import gradients
from pg_methods.policies import CategoricalPolicy, GaussianPolicy
from pg_methods.networks import MLP_factory
from pg_methods.objectives import PolicyGradientObjective
import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
import logging

CG_ITERS = 1
BATCH_SIZE = 5

"""Helper functions."""
def get_distribution_discrete(fn_approximator, state, detach=False):
  policy_log_probs = fn_approximator(state)
  probs = F.softmax(policy_log_probs, dim=1)
  if detach:
    probs = probs.detach()
  stochastic_policy = Categorical(probs)
  return stochastic_policy

def get_distribution_gaussian(fn_approximator, state, detach=False):
  policy_mu, policy_sigma = fn_approximator(state)
  policy_sigma = F.softplus(policy_sigma)
  if detach:
    policy_mu = policy_mu.detach()
    policy_sigma = policy_sigma.detach()
  stochastic_policy = Normal(policy_mu, policy_sigma)
  return stochastic_policy

def compute_hessian_vector_product(policy, observations, vector, dist_get, regularizer=1e-9):
  vector = torch.from_numpy(vector).float()
  policy.zero_grad()
  old_policy = dist_get(copy.deepcopy(policy).fn_approximator, observations, detach=True)
  new_policy = dist_get(policy.fn_approximator, observations)
  mean_kl = torch.mean(torch.distributions.kl_divergence(new_policy, old_policy))
  grads = torch.autograd.grad(mean_kl, policy.parameters(), create_graph=True)
  flat_grad = parameters_to_vector(grads)
  h = torch.sum(flat_grad * vector)
  hvp = torch.autograd.grad(h, policy.parameters())
  hvp_flat = np.concatenate([g.contiguous().view(-1).numpy() for g in hvp])
  return hvp_flat + regularizer * vector


def run_algorithm(MODE, alpha, baseline):
    gamma = 0.99
    n_episodes = 500
    logging.info('Mode: %s', MODE)

    objective = PolicyGradientObjective()
    # env = parallelized_gym.SubprocVecEnv([lambda : PointEnv() for _ in range(5)])
    # dist_get = get_distribution_gaussian

    env_name = 'MountainCar-v0'
    # env_name = 'CartPole-v0'
    env = interfaces.make_parallelized_gym_env(env_name,
                                               seed=int(time.time()),
                                               n_workers=BATCH_SIZE)
    dist_get = get_distribution_discrete

    fn_approximator, policy = experiment.setup_policy(
        env, hidden_non_linearity=torch.nn.ReLU, hidden_sizes=[16, 16])
    if baseline == 'moving_average':
        baseline = MovingAverageBaseline(0.99)
    elif baseline == 'function_approximator':
        input_size = env.observation_space_info['shape'][0]
        value_function = MLP_factory(input_size,
                                     hidden_sizes=[16, 16],
                                     output_size=1,
                                     hidden_non_linearity=torch.nn.ReLU)
        optimizer = torch.optim.RMSprop(value_function .parameters(), lr=0.001)
        baseline = FunctionApproximatorBaseline(value_function, optimizer)
    else:
        raise ValueError('Unknown baseline.')

    accum_rewards = []
    for i in range(n_episodes):
        trajectories = obtain_trajectories(env, policy, 200,
                                           reset=True, value_function=baseline)
        trajectories.torchify()
        returns = gradients.calculate_returns(trajectories.rewards, gamma, trajectories.masks)
        advantages = returns - trajectories.values
        baseline_loss = baseline.update_baseline(trajectories, returns)
        loss = objective(advantages, trajectories)
        policy.zero_grad()
        vpg_grad = torch.autograd.grad(loss, policy.parameters(), create_graph=True)
        vpg_grad = parameters_to_vector(vpg_grad).detach().numpy()

        curr_params = parameters_to_vector(policy.parameters())
        if MODE == 'npg':
            #   print('vpg_grad',vpg_grad)
            # Last state is just the state after getting our done so we leave it out.
            states_to_process = trajectories.states[:-1]
            traj_len, batch_size, state_shape = states_to_process.size()
            states_to_process = states_to_process.view(traj_len * batch_size, state_shape)

            def hvp_fn(vector):
                return compute_hessian_vector_product(policy, states_to_process, vector, dist_get)

            npg_grad = conjugate_gradient_algorithm(hvp_fn,
                                                    vpg_grad,
                                                    # x_0=vpg_grad.copy(),
                                                    cg_iters=CG_ITERS)

            #   if alpha is not None:
            #   n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
            #             else:
            #               n_step_size = self.n_step_size
            eff_alpha = np.sqrt(np.abs(alpha / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))
            if np.allclose(npg_grad, vpg_grad): raise ValueError('No change in npg, vpg')
            new_params = curr_params - eff_alpha * torch.from_numpy(npg_grad)
            accum_rewards_npg = accum_rewards
        elif MODE == 'vpg':
            new_params = curr_params - alpha * torch.from_numpy(vpg_grad)
            accum_rewards_vpg = accum_rewards
        else:
            raise ValueError('Unkown algorithm')
        vector_to_parameters(new_params, policy.parameters())
        reward_summary = torch.sum(trajectories.rewards * trajectories.masks.float(), dim=0)
        #   print(reward_summary.mean())
        accum_rewards.append(reward_summary.mean())

    return accum_rewards


if __name__ == '__main__':
    # Good arguments for cartpole
    # baseline = 'moving_average'
    baseline = 'function_approximator'
    # logging.set_verbosity(logging.INFO)
    logging.info('Baseline being used %s', baseline)

    accum_rewards_vpg = run_algorithm('vpg', 0.0006, baseline)
    accum_rewards_npg = run_algorithm('npg', 0.0007, baseline)

    plt.plot(accum_rewards_vpg, label='vpg')
    plt.plot(accum_rewards_npg, color='r', label='npg')
    torch.save({'vpg'.format(baseline):accum_rewards_vpg,
                'npg'.format(baseline): accum_rewards_npg},
               '{}.pyt'.format(baseline))
    plt.xlabel('Updates')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    plt.hold()