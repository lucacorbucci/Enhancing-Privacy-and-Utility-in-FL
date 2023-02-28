class Performances:
    """This class is used to store the performances of a node."""

    def __init__(
        self,
        node_name: str,
        epochs: list,
        loss_list: list[float],
        accuracy_list: list[float],
        loss: float | None,
        accuracy: float | None,
        message_counter: int,
        epsilon_list: list[float] | None,
    ) -> None:
        """Initializes the object Performances.

        Args:
            node_name (str): the name of the node
            epochs (int): the number of epochs
            loss_list (List[float]): the list of the losses
            accuracy_list (List[float]): the list of the accuracies
            loss (float): loss of the last epoch
            accuracy (float): accuracy of the last epoch
            message_counter (int): the number of exchanged messages
            epsilon_list (List[float]): the list of the epsilon values
        """
        self.node_name = node_name
        self.epochs = epochs
        self.loss_list = loss_list
        self.accuracy_list = accuracy_list
        self.loss = loss
        self.accuracy = accuracy
        self.message_counter = message_counter
        self.epsilon_list = epsilon_list

    def __str__(self) -> str:
        return f"node_name: {self.node_name}, epochs: {self.epochs}, loss_list: \
            {self.loss_list}, accuracy_list: {self.accuracy_list}, loss: {self.loss}, \
                accuracy: {self.accuracy}, exchanged messages: {self.message_counter}, \
                    epsilon list: {self.epsilon_list}"
