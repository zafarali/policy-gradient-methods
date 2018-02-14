"""
Utilities for actor critic implementations.
Mostly to allow us to use shared and unshared
architectures interchangeably and not run two
forward passes
"""
import torch

class ActorCriticController(object):
    def __init__(self, network):
        self.state = None
        self.network = network
        self.shared = self.network._pg_ac_shared

    def execute_updates(self):


    def detect_state_change(self, state):
        """
        Returns True if the state has changed.
        :param state: N x |S| representing states
        :return:
        """

        ## TODO: check if this is computationally efficient?
        ## maybe there is a simple hash that we can do?
        ## (can confirm hash(state) hash(self.state) will be different even with the same data
        if self.state is None:
            return True
        return not(torch.sum(torch.eq(state, self.state)) == state.size()[0] * state.size()[1])

    def run_forward_pass(self, state, update_state=False):
        value, action, log_probs = self.network(state)
        self.value = value
        self.action = action
        self.log_probs = log_probs
        if update_state: self.state = state
        return self.value, self.action, self.log_probs

    def policy(self, state):
        if self.detect_state_change(state):
            self.run_forward_pass(state, update_state=True)

        return self.action, self.log_probs

    def value_fn(self, state):
        if self.detect_state_change(state):
            self.run_forward_pass(state, update_state=True)
        return self.value