import torch.nn as nn

class SharedActorCritic(nn.Module):
    def __init__(self,
                shared_architecture,
                action_space_size,
                shared_architecture_out_size,
                h_common2actor,
                h_common2value,
                policy_class,
                hidden_non_linearity=nn.ReLU):
        """
        Implements a shared Actor Critic achitecture.
        where the main trunk of the network is shared
        :param shared_architecture: the shared trunk of the two networks
        :param action_space: the size of the action space
        :param shared_architecture_out_size: the size of the output layer 
                            in the shared archtecture (@TODO: find out a way to automatically figure this out)
        :param h_common2actor: the hidden layer between the common architecture and the actor
        :param h_common2value: the hidden layer between the common architecture and the value function
        :param policy_class: a pg_methods.utils.policies.Policy class that has not been instantiated yet
        :param hidden_non_linearity: the nonlinearity to use in the branched out approximator
        """
        super().__init__()
        # self.common_architecture = shared_architecture
        self.actor_fn_approximator = nn.Sequential(
                                        shared_architecture,
                                        hidden_non_linearity(),
                                        nn.Linear(shared_architecture_out_size, h_common2actor),
                                        hidden_non_linearity(),
                                        nn.Linear(h_common2actor, action_space_size))
        self.actor = policy_class(self.actor_fn_approximator)
        self.value_fn_approximator = nn.Sequential(
                                        shared_architecture,
                                        hidden_non_linearity(),
                                        nn.Linear(shared_architecture_out_size, h_common2value),
                                        hidden_non_linearity(),
                                        nn.Linear(h_common2value, 1))

    def forward(self, state):
        action, log_probs = self.actor(state)
        value = self.value_fn_approximator(state)
        return value, action, log_probs

    def evaluate_actor_log_prob(self, states, actions):
        return self.actor.log_prob(states, actions)
