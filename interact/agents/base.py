"""Core components of the interact package.

Author: Ryan Strauss
"""

from abc import ABC, abstractmethod

from gym import Env


class Agent(ABC):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment by first observing the state of the environment. Based on this
    observation the agent changes the environment by performing an action.

    Args:
        env: The environment that the agent is bound to.
        load_path: Optional path to a checkpoint that will be used to initialize the agent's network(s).
    """

    def __init__(self, *, env, load_path=None):
        assert isinstance(env, Env), 'env must be an instance of gym.Env'

        self._env = env

        if load_path is not None:
            self.load(load_path)

    @abstractmethod
    def learn(self, *, total_timesteps, logger, log_interval=100, save_interval=None):
        """Executes the agent's training process.

        Args:
            total_timesteps: The total number of timesteps in the environment that the agent will train for.
            logger: The logger to use for saving training information.
            log_interval: The period (in updates) at which TensorBoard logs will be saved.
            save_interval: The period (in updates) at which network weights will be saved.

        Returns:
            None
        """
        pass

    @abstractmethod
    def act(self, observation):
        """Selects the action to be taken based on an observation from an environment.

        Args:
            observation: An observation of an environment state.

        Returns:
            The action prescribed by the agent to be taken in the given state.
        """
        pass

    @abstractmethod
    def load(self, path):
        """Loads saved weights into the agent's network(s).

        Args:
            path: Path to a checkpoint containing the desired weights.

        Returns:
            None
        """
        pass

    @abstractmethod
    def save(self, path):
        """Saves agent's weights to a file.

        Args:
            path: Location to save weights.

        Returns:
            None
        """
        pass
