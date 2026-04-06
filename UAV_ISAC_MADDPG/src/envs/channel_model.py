import warnings
from src.envs.models.communication_model import CommunicationModel as ChannelModel

warnings.warn(
    "src/envs/channel_model.py is deprecated, use src/envs/models/communication_model.py instead", 
    DeprecationWarning, 
    stacklevel=2
)

__all__ = ["ChannelModel"]