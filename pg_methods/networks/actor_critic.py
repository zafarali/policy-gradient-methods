"""
These are wrappers to allow us to
use a common interface for Actor Critics
whether the architecture is shared or not.
It saves one forward pass computation in the shared case.
"""
import torch.nn as nn


# TODO: check if this "ActorCritic" abstraction is necessary
# and calling backward() doesn't mix the loss functions
# when registered to a module?

class ActorCritic(object):
    """
    Unshared Actor Critic architecture.
    """
    _pg_ac_shared = False
    def __init__(self,
                 actor,
                 critic):
        self.actor = actor
        self.critic = critic

    def __call__(self, state):
        # to mimic the PyTorch object.
        return self.forward(state)

    def forward(self, state):
        action, log_prob = self.actor(state)
        value_estimate = self.critic(state)
        return value_estimate, action, log_prob

    def cuda(self):
        self.actor = self.actor.cuda()
        self.critic = self.critic.cuda()

    def cpu(self):
        self.actor = self.actor.cpu()
        self.critic = self.critic.cpu()

    def __repr__(self):
        # taken from the NN module class:
        tmpstr = self.__class__.__name__ + '(\n'
        tmpstr = tmpstr + '  (actor): ' + self.actor.__repr__() + '\n'
        tmpstr = tmpstr + '  (critic): ' + self.critic.__repr__() + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


class SharedActorCritic(nn.Module):
    _pg_ac_shared = True
    def __init__(self,
                 common_architecture,
                 actor,
                 critic):
        super().__init__()
        self.common_architecture = common_architecture
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        state_representation = self.common_architecture(state)
        action, log_prob = self.actor(state_representation)
        value_estimate = self.critic(state_representation)
        return value_estimate, action, log_prob
