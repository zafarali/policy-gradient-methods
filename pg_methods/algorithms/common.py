from pg_methods.utils.objectives import PolicyGradientObjective

class Algorithm(object):
    def __init__(self, environment, policy, objective=PolicyGradientObjective(), logger=None, use_cuda=False):
        self.environment = environment
        if use_cuda:
            self.policy = policy.cuda()
        else:
            self.policy = policy
        self.use_cuda = use_cuda
        self.logger = logger
        self.objective = objective

    def log(self, **kwargs):
        if self.logger is not None:
            self.logger(**kwargs)
        
    def _run_step(self):
        raise NotImplementedError('Algorithm not implemented')

    def _process_episode_outputs(self, outputs):
        """
        Process the output of an run on an episode
        This will return a reward and maybe loss
        for printing and saving
        """
        rewards = outputs
        return rewards

    def _stopping_criterion(self, episode_reward):
        """
        This allows for an early stopping criterion
        """
        return False

    def run(self, n_episodes):
        for i in range(n_episodes):
            self._run_step()

            if self._stopping_criterion():
                break