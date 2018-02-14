from .. import gradients

"""
Abstract classes to calculate the objective of an algorithm
"""
class Objective(object):
    _calls_init = False
    def __init__(self):
        self.calls = 0
        self._calls_init = True

    def _increment_count(self):
        # counts the number of times this object has been called
        # can be useful in the future to have a scheduled hyperparameter.
        if not self._calls_init:
            self.calls = 0

        self.calls += 1

    def _calculate(self, advantages, trajectory):
        """
        Convenience function that does the internal calculations
        """
        raise NotImplementedError('The objective has not implemented the _calculate method')

    def __call__(self, advantages, trajectory):
        """
        Calculates the objective to be used in backward()
        :param advantages: the advantage to be used in the policy gradient
        :param trajectory: a data.storage.MultiTrajectory
                           or data.storage.Trajectory object
                           which has been `torchify()`'ed
        :return:
        """
        self._increment_count()
        return self._calculate(advantages, trajectory)



class PolicyGradientObjective(Objective):
    def __init__(self, entropy=0, time_mean=False):
        """
        Calculates the policy gradient objective:

        E[logprob * (R - baseline)]

        Additional options include:
            1. taking the time mean (instead of adding terms over time)
            2.
        :param entropy: the value of the entropy regularization
        :param time_mean:
        """
        self.entropy = entropy
        self.time_mean = time_mean

    def _calculate(self, advantages, trajectory):


        loss = gradients.calculate_policy_gradient_terms(trajectory.log_probs, advantages)

        print(loss)

        loss = loss.sum(dim=0)
        if self.time_mean:
            loss = loss / trajectory.masks.sum(dim=0)


        if self.entropy is not False:
            entropy = gradients.get_entropy(trajectory.log_probs).sum(dim=0)
            if self.time_mean:
                entropy = entropy / trajectory.masks.sum(dim=0)

            loss -= self.entropy * entropy


        loss = loss.mean()

        return loss


class NaturalPolicyGradientObjective(Objective):
    def __init__(self):
        raise NotImplementedError('Not implemented yet')