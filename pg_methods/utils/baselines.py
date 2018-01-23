"""
Implements some common baselines
"""
import numpy as np

class Baseline(object):
    def update(self, reward):
        raise NotImplementedError('need to implement!')
    def __call__(self, state):
        return 0


class MovingAverageBaseline(Baseline):
    def __init__(self, beta):
        super().__init__()
        self.value = 0
        self.beta = beta
        self.initiated = False

    def update_baseline(self, rewards):
        if not self.initiated:
            self.value = np.array(rewards).mean()
            self.initiated = True
        else:
            self.value = self.value * (1-self.beta) + self.beta * np.array(rewards).mean()

    def __call__(self, state):
        return self.value

    def __str__(self):
        return 'MovingAverageBaseline({})'.format(self.beta)