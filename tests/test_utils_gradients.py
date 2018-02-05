from pg_methods.utils.gradients import calculate_returns, calculate_policy_gradient_terms
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



