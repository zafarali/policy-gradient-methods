import torch.nn as nn

class ActorCritic(object):
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
