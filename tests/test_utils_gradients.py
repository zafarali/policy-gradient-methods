from pg_methods.utils.gradients import (calculate_returns,
                                        calculate_policy_gradient_terms,
                                        calculate_bootstrapped_returns)
import numpy as np
import torch

def test_calculate_returns():
    rewards = torch.FloatTensor([
                        [0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 1, 1],
                        [1, 1, 0, 1]])
    discount = 0.5
    masks = torch.FloatTensor([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 0],
                            [1, 1, 0, 0]])
    # test batch mode
    result = calculate_returns(rewards, discount, masks).numpy()

    expected = np.array([[0.125, 1.875, 0.25, 1.5],
                         [0.25, 1.75, 0.5, 1.0],
                         [0.5, 1.5, 1.0, 0.0],
                         [1.0, 1.0, 0.0, 0.0]])

    assert np.allclose(result, expected)

    # test unbatched mode

    rewards = torch.FloatTensor([[0], [0], [0], [1]])
    result = calculate_returns(rewards, discount).numpy()

    assert np.allclose(result, np.array([[0.125], [0.25], [0.5], [1.0]]))

def test_calculate_policy_gradient():
    probs = torch.FloatTensor([[0.5, 0.5, 0.15],
                               [0.5, 0.33, 0.44]])
    advantages = torch.FloatTensor([[0, 2, 1],
                                    [1, 1, 5]])
    log_probs = probs.log()

    # calculate the logprobs * adv by hand
    answers = (-np.log(0.5)*2 - np.log(0.33)*1) + (-np.log(0.5)*0 - np.log(0.5)*1) + (-np.log(0.15)*1 - np.log(0.44)*5)
    answers /= 3

    result = calculate_policy_gradient_terms(log_probs, advantages).data.sum(dim=0).mean()
    assert np.allclose(result, answers)


def test_calculate_policy_gradient_with_masks():

    # 3 agents, 2 time steps, one agent only takes one step.
    probs = torch.FloatTensor([[0.5, 0.5, 0.15],
                               [0.5, 0.33, 0.44]])
    advantages = torch.FloatTensor([[0, 2, 1],
                                    [1, 1, 5]])

    masks = torch.FloatTensor([[1, 1, 1],
                               [1, 1, 0]])
    log_probs = probs.log()

    # calculate the logprobs * adv by hand
    answers = (-np.log(0.5)*2 - np.log(0.33)*1) + (-np.log(0.5)*0 - np.log(0.5)*1) + (-np.log(0.15)*1)
    answers /= 3

    result = calculate_policy_gradient_terms(log_probs, advantages, masks).data.sum(dim=0).mean()
    assert np.allclose(result, answers)

def test_calculate_bootstrapped_returns():
    rewards = torch.FloatTensor([[5, 3, 4],
                                 [8, 2, 3],
                                 [9, -100, 1],
                                 [-100, -100, 5]])

    masks = torch.FloatTensor([[1, 1, 1],
                               [1, 1, 1],
                               [1, 0, 1],
                               [0, 0, 1]])

    value_estimates = torch.FloatTensor([[10, 2, 4],
                                         [8, 1, 5],
                                         [5, 0, 3],
                                         [3, 4, 1],
                                         [4, 4, 10]])

    latest_value_estimate = value_estimates[-1]
    returns = calculate_bootstrapped_returns(rewards, 0.5, latest_value_estimate, masks)

    agent_1_returns = [5 + 0.5*8 + 0.5**2 * 9,
                       8 + 0.5*9,
                       9,
                       0]

    agent_2_returns = [3 + 0.5*2,
                       2,
                       0,
                       0]

    agent_3_returns = [4 + 0.5*3 + 1*0.5**2 + 5*0.5**3 + 10*0.5**4,
                       3 + 0.5 + 5*0.5**2 + 10*0.5**3,
                       1 + 0.5*5 + 10*0.5**2,
                       5 + 0.5*10]

    expected_returns = np.array([agent_1_returns, agent_2_returns, agent_3_returns]).T
    print('expected:',expected_returns)
    print('got:',returns)
    assert np.allclose(returns.numpy(), expected_returns)


