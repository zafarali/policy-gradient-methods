from .. import gradients

class Advantage(object):
    def __init__(self):
        pass


    def _calculate(self, returns, values):
        """
        Convenience function that does the internal calculations
        """
        raise NotImplementedError('The advantage has not implemented the _calculate method')


    def __call__(self, returns, values):
        return self._calculate(returns, values)



class VanillaAdvantage(Advantage):
    def _calculate(self, returns, values):
        return returns - values