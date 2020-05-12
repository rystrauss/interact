from interact.common.runners import AbstractRunner


class Runner(AbstractRunner):

    def __init__(self, env, policy, nsteps, gamma):
        super().__init__(env, policy, nsteps)

        self.gamma = gamma

    def run(self):
        pass
