import gin as _gin
from gym.wrappers import AtariPreprocessing as _AtariPreprocessing

from .utils import make_env_fn

_gin.external_configurable(_AtariPreprocessing, denylist=["env"])
