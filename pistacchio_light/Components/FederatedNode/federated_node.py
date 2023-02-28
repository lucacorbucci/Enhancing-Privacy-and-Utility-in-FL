import sys
import time
from typing import Any, Mapping, TypeVar

from loguru import logger
from torch import Tensor, nn

from pistacchio_simulator.Exceptions.errors import NotYetInitializedServerChannelError
from pistacchio_simulator.Models.federated_model import FederatedModel
from pistacchio_simulator.Utils.communication_channel import CommunicationChannel
from pistacchio_simulator.Utils.end_messages import Message
from pistacchio_simulator.Utils.performances import Performances
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.weights import Weights


logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)

#TDestination = TypeVar("TDestination ", bound=Mapping[str, Tensor])


class FederatedNode:
    """FederatedNode is the component that can serve as an abstraction
    class for creating one federated node.
    """

    def __init__(
        self,
        node_id: str,
        preferences = None,
        configuration = None
        # server_channel: CommunicationChannel,
        # logging_queue: CommunicationChannel,
        # receiver_channel: CommunicationChannel | None = None,
        ) -> None:
        """Init the Federated Node with either preferences object or 
            configuration dictionary. 
        Args:
            node_id (str): id of the node
            preferences (Preferences): preferences object of the node that contains
                all the preferences for this node
            configuration (dict): configuration dictionary that contains all the
                preferences for this node.
            # logging_queue (CommunicationChannel): queue that is used to send back the
            #     performances of the node to the main thread.
        """
        # self.logging_queue = logging_queue
        # self.receiver_channel = (
        #     receiver_channel if receiver_channel else CommunicationChannel(name=node_id)
        # )
        # self.server_channel: CommunicationChannel | None = None
        self.federated_model = None
        self.message_counter = 0
        self.mixed = False

        assert preferences or configuration
        #TODO: Create initialization from preferences.
        if preferences:
            self.preferences = preferences
        else:
            pass
        
        self.mode = "federated"
        self.node_id = node_id
        # Print debug function [TO DELETE]
        print(f"Node {self.node_id} initialized")

    def send_weights_to_server(self, weights: Weights) -> None:
        """This function is used to send the weights of the nodes to the server.

        Args:
            weights (Weights): weights to be sent to the server
        Raises:
            ValueError: Raised when the server channel is not initialized
        """
        if self.server_channel:
            self.server_channel.send_data(weights)
        else:
            raise ValueError("Server channel not initialized")


    def add_server_channel(self, server_channel: CommunicationChannel) -> None:
        """This function adds the server channel to the sender thread.

        Args:
            server_channel (_type_): server channel
        """
        self.server_channel = server_channel


    def init_federated_model(self, model: nn.Module) -> FederatedModel:
        """Initialize the federated learning model.

        Args:
            model (_type_): _description_

        Returns
        -------
            FederatedModel: _description_
        """
        federated_model: FederatedModel = FederatedModel(
            dataset_name=self.preferences.dataset_name,
            node_name=self.node_id,
            preferences=self.preferences,
        )
        federated_model.init_model(net=model)

        return federated_model

 
    def local_training(
        self,
        differential_private_train: bool,
    ) -> dict:
        """_summary_.

        Args:
            differential_private_train (bool): _description_
            federated_model (FederatedModel): _description_

        Returns
        -------
            dict: _description_
        """
        epsilon = None
        if differential_private_train:
            (
                loss,
                accuracy,
                epsilon,
            ) = self.federated_model.train_with_differential_privacy()
        else:
            loss, accuracy = self.federated_model.train()
        return {"loss": loss, "accuracy": accuracy, "epsilon": epsilon}


    def send_and_receive_weights_with_server(
        self,
        federated_model: FederatedModel,
        metrics: dict,
        results: dict | None = None,
    ) -> Any:
        """Send weights to the server and receive the
        updated weights from the server.

        Args:
            federated_model (FederatedModel): Federated model
            metrics (dict): metrics computed on the node (loss, accuracy, epsilon)

        Returns
        -------
            _type_: weights received from the server
        """
        # Create the Weights object that we will send to the server
        weights = Weights(
            weights=federated_model.get_weights(),
            sender=self.node_id,
            epsilon=metrics["epsilon"],
            results=results,
        )
        # Send weights to the server
        self.send_weights_to_server(weights)
        self.message_counter += 1
        # Receive the updated weights from the server
        received_weights: Weights | Message = self.receive_data_from_server()
        self.message_counter += 1

        return received_weights

    
    def send_performances(self, performances: dict[str, Performances]) -> None:
        """This function is used to send the performances of
        the node to the server.

        Args:
            performances (Performances): _description_
        """
        if self.server_channel:
            self.server_channel.send_data(performances)
        else:
            raise NotYetInitializedServerChannelError


    def compute_performances(
        self,
        loss_list: list,
        accuracy_list: list,
        phase: str,
        message_counter: int,
        epsilon_list: list | None,
    ) -> dict:
        """This function is used to compute the performances
        of the node. In particulare we conside the list of
        loss, accuracy and epsilon computed during the
        local training on the node.

        Args:
            loss_list (List): list of loss computed during the local training
            accuracy_list (List): list of accuracy computed during the local training
            phase (str): Phase of the training (P2P or server)
            message_counter (int): count of the exchanged messages
            epsilon_list (List, optional): list of epsilon computed
                during the local training. Defaults to None.

        Returns
            Performances: Performance object of the node
        """
        epochs = range(
            1,
            self.preferences.server_config["num_communication_round_with_server"] + 1,
        )

        performances = {}
        performances[phase] = Performances(
            node_name=self.node_id,
            epochs=list(epochs),
            loss_list=loss_list,
            accuracy_list=accuracy_list,
            loss=None,
            accuracy=None,
            message_counter=message_counter,
            epsilon_list=epsilon_list,
        )

        return performances


    def receive_starting_model_from_server(
        self,
        federated_model: FederatedModel,
    ) -> None:
        """This function is used to receive the starting model
        from the server so that all the nodes start the federated training
        from the same random weights.

        Args:
            federated_model (FederatedModel): The federated model we want
            to initialize with the received weights
        """
        received_weights = self.receive_data_from_server()
        federated_model.update_weights(received_weights)


    def start_node(self, model: nn.Module) -> None:
        """This method implements all the logic of the federated node.
        It starts the training of the model and then sends the weights to the
        server.
        Then, after the end of the training, it sends the performances of the
        node to the main thread.

        Args:
            model (_type_): Model that we want to use during the federated learning
        """
        logger.debug(f"Starting node {self.node_id}")
        self.federated_model = self.init_federated_model(model)
        
        # self.receive_starting_model_from_server(federated_model=federated_model)
        # logger.debug(f"Node {self.node_id} received starting model from server")
        #differential_private_train = self.preferences.server_config[
            #"differential_privacy_server"
        #]
        # Initialize differential privacy if needed
        # if differential_private_train:
        #     self.federated_model.init_differential_privacy(phase=Phase.SERVER, node_id=self.node_id)
        #     logger.debug(f"Node {self.node_id} initialized differential privacy")
        
        logger.debug(f"Node {self.node_id} started")

     
    def train_local_model(
        self,
        # results: dict | None = None,
    ) -> tuple[list[float], list[float], list[float]]:
        """This function starts the server phase of the federated learning.
        In particular, it trains the model locally and then sends the weights.
        Then the updated weights are received and used to update
        the local model.

        Args:
            federated_model (FederatedModel): _description_

        Returns
        -------
            Tuple[List[float], List[float], List[float]]: _description_
        """
        logger.debug(f"Starting training on node {self.node_id}")
        loss_list: list[float] = []
        accuracy_list: list[float] = []
        epsilon_list: list[float] = []

        local_epochs = self.preferences.server_config[
            "local_training_epochs_with_server"
        ]
        differential_private_train = self.preferences.server_config[
            "differential_privacy_server"
        ]

        for _ in range(local_epochs):
            metrics = self.local_training(
                differential_private_train,
            )
            loss_list.append(metrics["loss"])
            accuracy_list.append(metrics["accuracy"])
            if metrics.get("epsilon", None):
                epsilon_list.append(metrics["epsilon"])
        
        logger.debug("2")
        return Weights(
            weights=self.federated_model.get_weights(),
            sender=self.node_id,
            epsilon=metrics["epsilon"],
        )

        received_weights = self.send_and_receive_weights_with_server(
            federated_model=federated_model,
            metrics=metrics,
            results=results,
        )

        #Update the weights of the model
        federated_model.update_weights(received_weights)

        return loss_list, accuracy_list, epsilon_list,
