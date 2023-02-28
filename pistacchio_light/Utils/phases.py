from enum import Enum


class Phase(Enum):
    """Enum class for types of phases of the training."""

    P2P = 1
    SERVER = 2
    MIXED = 3
