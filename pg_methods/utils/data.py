import sys
from collections import namedtuple

# is this efficient?
Transition = namedtuple('Transition', 'state action reward next_state')

class Trajectory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.baselines = []
        self.log_probs = []
        self.transitions = []

    def append(self, state_t, action, reward, baseline, log_prob, state_tp1):
        self.transitions.append(Transition(state_t, action, reward, state_tp1))
        
        if len(self.states) == 0:
            self.states.append(state_t)

        self.states.append(state_tp1)
        self.actions.append(action)
        self.rewards.append(reward)
        self.baselines.append(baseline)
        self.log_probs.append(log_prob)

    def __str__(self):
        return "Trajectory({})".format(self.transitions)


def obtain_trajectory(environment,
                      policy,
                      state_processor,
                      action_processor,
                      baseline_function=None,
                      max_steps=sys.maxsize,
                      verbose=False):
    """
    Obtain a trajectory by executing a policy in an environment
    """
    trajectory = Trajectory()

    state_t = environment.reset()

    if verbose: print("Start state: {}".format(state))

    state_t = state_processor.state2pytorch(state_t)

    for t in range(max_steps):
        action, log_prob = policy(state_t)

        action = action_processor.pytorch2gym(action)

        if verbose: print('Action taken: {}'.format(action))
        
        state_tp1, reward, done, info = environment.step(action)
        
        if verbose: print('State: {}, reward: {}, done {}'.format(state, reward, done))
        
        baseline = baseline_function(state_t) if baseline_function is not None else 0
        trajectory.append(state_processor.pytorch2state(state_t), action, reward, baseline, log_prob, state_tp1)

        state_t = state_processor.state2pytorch(state_tp1)

        if done:
            break

    return trajectory