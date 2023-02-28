from enum import Enum


class SplitTypes(Enum):
    """Enum for the different types of splits."""

    PURE = 1
    STRATIFIED = 2
    PERCENTAGE = 3
