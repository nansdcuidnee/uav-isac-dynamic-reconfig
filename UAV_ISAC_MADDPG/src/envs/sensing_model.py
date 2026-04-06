import warnings
from src.envs.models.sensing_model import SensingModel

warnings.warn(
    "src/envs/sensing_model.py is deprecated, use src/envs/models/sensing_model.py instead", 
    DeprecationWarning, 
    stacklevel=2
)

__all__ = ["SensingModel"]