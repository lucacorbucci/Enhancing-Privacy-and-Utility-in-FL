import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, TypeVar

import dill
import numpy as np
import torch
from loguru import logger
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pistacchio_simulator.Exceptions.errors import (
    InvalidDatasetNameError,
    NotYetInitializedFederatedLearningError,
    NotYetInitializedPreferencesError,
    NotYetInitializedServerChannelError,
)
from pistacchio_simulator.Models.federated_model import FederatedModel
from pistacchio_simulator.Utils.communication_channel import CommunicationChannel
from pistacchio_simulator.Utils.data_loader import DataLoader
from pistacchio_simulator.Utils.end_messages import Message
from pistacchio_simulator.Utils.learning import Learning
from pistacchio_simulator.Utils.performances import Performances
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.weights import Weights
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import Tensor, nn, optim

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)

TDestination = TypeVar("TDestination ", bound=Mapping[str, Tensor])


class FederatedNode:
    """FederatedNode is the component that can be used for the
    classic Federated Learning.
    It trains the model locally and then sends the weights to the server.
    """

    def __init__(
        self,
        node_id: str,
        cluster_id: str,
        node_name: str,
        preferences: Preferences,
        phase: Phase,
        model: FederatedModel,
    ) -> None:
        """Init the Federated Node.

        Args:
            node_id (str): id of the node
            preferences (Preferences): preferences object of the node that contains
                all the preferences for this node
            # logging_queue (CommunicationChannel): queue that is used to send back the
            #     performances of the node to the main thread.
        """
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.preferences = preferences
        self.mode = "federated"
        self.mixed = False
        self.message_counter = 0
        self.federated_model = model
        self.node_name = node_name
        self.node_folder_path = (
            f"../data/{self.preferences.dataset}/nodes_data/{self.node_name}/"
        )
        self.load_data()
        self.load_privacy_engine()
        self.init_differential_privacy(phase=phase)

    def load_data(self):
        if self.preferences.public_private_experiment and self.phase == Phase.P2P:
            self.train_loader = DataLoader().load_splitted_dataset(
                f"../data/{self.preferences.dataset}/federated_data/{self.node_name}_public_train.pt",
            )
        elif self.preferences.public_private_experiment and self.phase == Phase.SERVER:
            self.train_loader = DataLoader().load_splitted_dataset(
                f"../data/{self.preferences.dataset}/federated_data/{self.node_name}_private_train.pt",
            )
        else:
            self.train_loader = DataLoader().load_splitted_dataset(
                f"../data/{self.preferences.dataset}/federated_data/{self.node_name}_train.pt",
            )
        self.test_set = DataLoader().load_splitted_dataset(
            f"../data/{self.preferences.dataset}/federated_data/{self.node_name}_test.pt",
        )

    def load_privacy_engine(self):
        self.privacy_engine = None

        # If we already used this client we need to load the state regarding
        # the private model
        if os.path.exists(f"{self.node_folder_path}privacy_engine.pkl"):
            logger.info(f"Loading Privacy Engine on node {self.node_name}")
            with open(f"{self.node_folder_path}privacy_engine.pkl", "rb") as file:
                self.privacy_engine = dill.load(file)

        if not self.privacy_engine:
            self.privacy_engine = PrivacyEngine(accountant="rdp")

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

    def load_data(
        self,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load training and test dataset.

        Returns
        -------
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: training and test set

        Raises
        ------
            Exception: Preference is not initialized
        """

        if self.preferences:
            batch_size = self.preferences.hyperparameters_config.batch_size
            if self.preferences.public_private_experiment:
                self.trainloader_private = torch.utils.data.DataLoader(
                    self.server_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                )
                self.trainloader_public = torch.utils.data.DataLoader(
                    self.p2p_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                )
                self.trainloader = None
            else:
                self.trainloader = torch.utils.data.DataLoader(
                    self.training_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                )

            self.testloader = torch.utils.data.DataLoader(
                self.test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
            )

            if self.preferences and self.preferences.debug:
                if self.trainloader:
                    self.print_data_stats(self.trainloader)
                else:
                    # One call is enough because we are summing
                    # the size of the two datasets
                    self.print_data_stats(self.trainloader_public)
        else:
            raise NotYetInitializedPreferencesError

    def print_data_stats(self, trainloader: torch.utils.data.DataLoader) -> None:
        """Debug function used to print stats about the loaded datasets.

        Args:
            trainloader (torch.utils.data.DataLoader): training set
        """
        if self.training_set:
            num_examples = {
                "trainset": len(self.training_set),
                "testset": len(self.test_set),
            }
        else:
            num_examples = {
                "trainset": len(self.server_dataset) + len(self.p2p_dataset),
                "testset": len(self.test_set),
            }
        targets = []
        for _, data in enumerate(trainloader, 0):
            targets.append(data[1])
        targets = [item.item() for sublist in targets for item in sublist]
        logger.info(f"{self.node_name}, {Counter(targets)}")
        logger.info(f"Training set size: {num_examples['trainset']}")
        logger.info(f"Test set size: {num_examples['testset']}")

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
            self.preferences.server_config.fl_rounds + 1,
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

    def train_local_model(
        self,
        phase: Phase,
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

        local_epochs = self.preferences.server_config.local_training_epochs
        differential_private_train = self.preferences.server_config.differential_privacy
        epsilon = None

        for _ in range(local_epochs):
            metrics = self.local_training(
                differential_private_train,
                phase=phase,
            )
            (
                loss,
                accuracy,
                epsilon,
                noise_multiplier,
            ) = Learning.train(phase=phase)

            metrics = {"loss": loss, "accuracy": accuracy, "epsilon": epsilon}

            loss_list.append(metrics["loss"])
            accuracy_list.append(metrics["accuracy"])
            if metrics.get("epsilon", None):
                epsilon_list.append(metrics["epsilon"])

        # We need to store the state of the privacy engine and all the
        # details about the private training

        Path.mkdir(
            self.node_folder_path,
            parents=True,
        )
        with open(f"{self.node_folder_path}/privacy_engine.pkl", "wb") as f:
            dill.dump(self.privacy_engine.accountant, f)

        return (
            Weights(
                weights=self.federated_model.get_weights(),
                sender=self.node_id,
                epsilon=metrics["epsilon"],
            ),
            metrics,
        )

    def init_differential_privacy(self, phase: Phase):
        epsilon = None
        noise_multiplier = None
        clipping = self.preferences.hyperparameters_config.max_grad_norm
        epsilon = (
            self.preferences.p2p_config.epsilon
            if phase == Phase.P2P and self.preferences.p2p_config.differential_privacy
            else self.preferences.server_config.epsilon
        )
        noise_multiplier = (
            self.preferences.p2p_config.noise_multiplier
            if phase == Phase.P2P and self.preferences.p2p_config.differential_privacy
            else self.preferences.server_config.noise_multiplier
        )

        train_loader = (
            self.trainloader_public
            if phase == Phase.P2P and self.preferences.p2p_config.differential_privacy
            else self.trainloader_private
        )
        if epsilon:
            (
                self.optimizer,
                self.train_loader,
                self.privacy_engine,
            ) = self.federated_model.init_privacy_with_epsilon(
                phase=phase,
                epsilon=epsilon,
                clipping=clipping,
                train_loader=train_loader,
                privacy_engine=self.privacy_engine,
                optimizer=self.optimizer,
                epochs=self.preferences.p2p_config.local_training_epochs
                if phase.P2P
                else self.preferences.server_config.local_training_epochs,
                delta=self.preferences.hyperparameters_config.delta,
            )
        elif noise_multiplier:
            (
                self.optimizer,
                self.train_loader,
                self.privacy_engine,
            ) = self.federated_model.init_privacy_with_epsilon(
                phase=phase,
                noise_multiplier=epsilon,
                clipping=clipping,
                train_loader=train_loader,
                privacy_engine=self.privacy_engine,
                optimizer=self.optimizer,
            )
