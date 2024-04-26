from enum import Enum

class LayerMode(Enum):
    LAYER_MIX = 1
    CHANNEL_MIX = 2
    LEGENCY = 3

class TextGenMode(Enum):
    SEQUENCE = 1
    TOKEN = 2