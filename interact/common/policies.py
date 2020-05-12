from abc import ABC, abstractmethod
from enum import Enum, auto

from gym.spaces import Discrete


class PolicyType(Enum):
    CATEGORICAL = auto()
    CONTINUOUS = auto()


class Policy(ABC):

    def __init__(self, ac_space, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ac_space = ac_space
        self._type = PolicyType.CATEGORICAL if isinstance(self.ac_space, Discrete) else PolicyType.CONTINUOUS

    @property
    def type(self) -> PolicyType:
        return self._type

    @abstractmethod
    def step(self, obs):
        """Returns the policy for a single step.

        Args:
            obs: The current observation of the environment.

        Returns:
            (actions, values, neglogp)
        """
        pass

    @abstractmethod
    def load_weights(self, path):
        pass

    @abstractmethod
    def save_weights(self, path):
        pass


class ActorCriticPolicy(Policy, ABC):

    def __init__(self, ac_space, *args, **kwargs):
        super().__init__(ac_space, *args, **kwargs)

        self._policy = None
        self._value_fn = None
