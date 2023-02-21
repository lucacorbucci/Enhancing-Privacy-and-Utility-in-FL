from typing import Mapping, TypeVar

from torch import Tensor


TDestination = TypeVar("TDestination", bound=Mapping[str, Tensor])


class Weights:
    """This class defines the weights of the model
    trained on a node.
    """

    def __init__(
        self,
        weights: TDestination,
        sender: str,
        epsilon: float = -1,
        results: dict | None = None,
    ) -> None:
        """This function initializes the weights object.

        Args:
            weights (TDestination): the weights of the model

            sender (str): the name of the node that sent the weights
        """
        self.weights = weights
        self.sender = sender
        self.epsilon = epsilon
        self.results = results
