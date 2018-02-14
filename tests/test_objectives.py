import torch
import numpy as np
from torch.autograd import Variable
from pg_methods.utils.objectives import PolicyGradientObjective
from pg_methods.utils.data.collectors import MultiTrajectory

def test_policy_gradient_objective():
    traj = MultiTrajectory(2)

    advantages = torch.FloatTensor([[0, 1],# agent 1 got reward 0, agent 2 got reward 1 at time step 1
                                   [2, 1],
                                   [1, 5]]).unsqueeze(-1)

    # this indicates that agent 1 action had prob of 0.5
    # and agent 2 had 0.33 at the second time step
    for log_probs in [(0.5, 0.5), (0.5, 0.33), (0.15, 0.44)]:
        # traj object is built over time, so each increment adds 1 to the time dimension
        # therefore this loop adds the respective log probs
        traj.append(torch.randn(2, 3),
                    torch.IntTensor([[0], [0]]),
                    torch.FloatTensor([[0], [0]]),
                    torch.FloatTensor([[0], [0]]),
                    torch.FloatTensor([log_probs[0], log_probs[1]]).log(),
                    torch.randn(2, 3),
                    torch.IntTensor([[False], [False]]))
    traj.torchify()
    print(traj.log_probs)
    print(advantages)
    objective = PolicyGradientObjective()
    loss = objective(advantages, traj)

    agent_1_logprob_x_adv = (-np.log(0.5)*0 - np.log(0.5) * 2 - np.log(0.15)*1)
    agent_2_logprob_x_adv = (-np.log(0.5)*1 - np.log(0.33) * 1 - np.log(0.44)*5)
    answers = (agent_1_logprob_x_adv + agent_2_logprob_x_adv)/2

    assert np.allclose(loss.data.numpy(), answers)
