"""
Implements some common baselines
"""
import torch
import numpy as np

class Baseline(object):
    def update_baseline(self, reward, advantages=None, values=None):
        raise NotImplementedError('need to implement!')

    def __call__(self, state):
        return 0


class MovingAverageBaseline(Baseline):
    def __init__(self, beta):
        super().__init__()
        self.value = 0
        self.beta = beta
        self.initiated = False

    def update_baseline(self, rewards, advantages=None, values=None):
        if not self.initiated:
            self.value = advantages.numpy().mean()
            self.initiated = True
        else:
            self.value = self.value * (1-self.beta) + self.beta * advantages.numpy().mean()

    def __call__(self, state):
        # first dim is the number o
        return self.value * torch.ones(state.size()[0])

    def __str__(self):
        return 'MovingAverageBaseline({})'.format(self.beta)