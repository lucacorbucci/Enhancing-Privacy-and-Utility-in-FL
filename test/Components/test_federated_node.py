import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn


sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from pistacchio_simulator.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_simulator.DataSplit.custom_dataset import MyDataset
from pistacchio_simulator.DataSplit.data_split import DataSplit
from pistacchio_simulator.DataSplit.storage_manager import StorageManager
from pistacchio_simulator.Exceptions.errors import NotYetInitializedServerChannelError
from pistacchio_simulator.Utils.communication_channel import CommunicationChannel
from pistacchio_simulator.Utils.end_messages import Message
from pistacchio_simulator.Utils.performances import Performances
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.task import Task
from pistacchio_simulator.Utils.weights import Weights


preferences = Preferences(
    dataset_name="mnist",
    task="federatedlearning",
    mode="classic",
    debug=False,
    wandb=False,
    save_model=False,
    data_split_config={
        "num_nodes": 5,
        "num_clusters": 1,
        "split_type": "percentage",
        "server_validation_set": "server_validation_split",
    },
    p2p_config={
        "local_training_epochs": 1,
        "diff_privacy_p2p": False,
        "num_communication_round_pre_training": [0],
    },
    server_config={
        "differential_privacy_server": False,
        "num_communication_round_with_server": 1,
    },
    hyperparameters={
        "batch_size": 32,
        "lr": 0.0001,
        "MAX_PHYSICAL_BATCH_SIZE": 128,
        "DELTA": 1e-5,
        "EPSILON": 0,
        "noise_multiplier": [0.5],
        "max_grad_norm": [10.0],
        "weight_decay": 0,
        "min_improvement": 0.001,
        "min_accuracy": 0.9,
        "patience": 1,
    },
)


@pytest.fixture(scope="session", autouse=True)
def pytest_configure() -> None:
    """_summary_.

    Args:
        config (_type_): _description_
    """
    samples = list(range(1000))
    targets = list(np.random.randint(0, 10, size=1000))
    y_tensor = torch.tensor(targets)
    y_tensor = y_tensor.float()
    x_tensor = torch.tensor(samples)
    x_tensor = x_tensor.float()

    mock_dataset = MyDataset(samples, targets)

    mock_dataset.classes = targets

    # Define the percentage configuration
    percentage_configuration = {
        "cluster_0": {0: 60, 1: 30, 2: 20, 3: 20},
        "cluster_1": {1: 70, 2: 40, 3: 20, 4: 20},
        "cluster_2": {2: 40, 3: 20, 4: 20, 5: 20},
        "cluster_3": {3: 40, 4: 20, 5: 20, 6: 30},
        "cluster_4": {4: 40, 5: 20, 6: 30, 7: 10},
        "cluster_5": {5: 40, 6: 20, 7: 30, 8: 30},
        "cluster_6": {6: 20, 7: 40, 8: 30, 9: 70},
        "cluster_7": {7: 20, 8: 40, 9: 30, 0: 40},
    }
    cluster_datasets, _ = DataSplit.percentage_split(
        dataset=mock_dataset,
        percentage_configuration=percentage_configuration,
        num_workers=8,
        task=Task("federatedlearning"),
    )

    names = [
        f"{node_id}_cluster_{cluster_id}"
        for cluster_id in range(8)
        for node_id in range(8)
    ]
    StorageManager.write_splitted_dataset(
        dataset_name="mnist",
        splitted_dataset=cluster_datasets,
        dataset_type="train_set",
        names=names,
    )
    StorageManager.write_splitted_dataset(
        dataset_name="mnist",
        splitted_dataset=cluster_datasets,
        dataset_type="test_set",
        names=names,
    )


class TestFederatedNode:
    """Test the FederatedNode class."""

    @staticmethod
    def init_test() -> FederatedNode:
        """Initialize a FederatedNode object for testing purposes.

        Returns
        -------
            _type_: a FederatedNode object
        """
        logging_queue = CommunicationChannel()
        federated_node = FederatedNode(
            node_id="0_cluster_0",
            preferences=preferences,
            logging_queue=logging_queue,
        )
        return federated_node

    @staticmethod
    def test_init() -> None:
        """Test the initialization of the FederatedNode class."""
        federated_node = TestFederatedNode.init_test()
        assert federated_node.node_id == "0_cluster_0"
        assert federated_node.preferences == preferences
        assert federated_node.logging_queue is not None
        assert federated_node.receiver_channel is not None

    @staticmethod
    def test_receive_data_from_server() -> None:
        """Test the receive_data_from_server method."""
        federated_node = TestFederatedNode.init_test()
        weights = torch.tensor([1, 2, 3])
        federated_node.receiver_channel.send_data(
            data=Weights(
                weights=weights,
                sender="Node 1",
                epsilon=0.05,
                results={"accuracy": 0.96},
            ),
        )
        received = federated_node.receive_data_from_server()
        assert torch.all(torch.eq(received, weights))

    @staticmethod
    def test_receive_stop_message_from_server() -> None:
        """Test the receive_data_from_server method."""
        federated_node = TestFederatedNode.init_test()

        federated_node.receiver_channel.send_data(data=Message.STOP)

        received = federated_node.receive_data_from_server()

        assert received == Message.STOP

    @staticmethod
    def test_receive_data_from_server_should_raise_an_error() -> None:
        """Test the receive_data_from_server method."""
        federated_node = TestFederatedNode.init_test()

        federated_node.receiver_channel.send_data(data={})

        result = federated_node.receive_data_from_server()
        assert result == Message.ERROR

    @staticmethod
    def test_send_weights_to_server() -> None:
        """Test the send_weights_to_server method."""
        federated_node = TestFederatedNode.init_test()
        data = Weights(
            weights=torch.tensor([1, 2, 3]),
            sender="Node 1",
            epsilon=0.05,
            results={"accuracy": 0.96},
        )
        server_channel = CommunicationChannel()
        federated_node.server_channel = server_channel
        federated_node.send_weights_to_server(weights=data)
        received_data = server_channel.receive_data()
        assert torch.all(torch.eq(received_data.weights, data.weights))
        assert received_data.sender == data.sender
        assert received_data.epsilon == data.epsilon
        assert received_data.results == data.results

    @staticmethod
    def test_send_weights_to_server_should_raise_an_exception() -> None:
        """Test the send_weights_to_server method."""
        federated_node = TestFederatedNode.init_test()
        weights = data = Weights(
            weights=torch.tensor([1, 2, 3]),
            sender="Node 1",
            epsilon=0.05,
            results={"accuracy": 0.96},
        )
        with pytest.raises(ValueError):
            federated_node.send_weights_to_server(weights=data)

    @staticmethod
    def test_add_server_channel() -> None:
        """Test the add_server_channel method."""
        federated_node = TestFederatedNode.init_test()
        server_channel = CommunicationChannel()
        federated_node.add_server_channel(server_channel)
        assert federated_node.server_channel == server_channel

    @staticmethod
    def test_init_federated_model() -> None:
        """Test the init_federated_model method."""
        federated_node = TestFederatedNode.init_test()
        model = nn.Linear(10, 10)
        federated_model = federated_node.init_federated_model(model)
        assert federated_model is not None
        assert federated_model.net is not None
        assert federated_model.net == model
        assert federated_model.node_name == "0_cluster_0"
        assert federated_model.preferences == preferences

    @staticmethod
    def test_get_communication_channel() -> None:
        """Test the get_communication_channel method."""
        federated_node = TestFederatedNode.init_test()
        channel = federated_node.get_communication_channel()
        assert channel is not None
        assert isinstance(channel, CommunicationChannel)

    @staticmethod
    def test_local_training_no_dp() -> None:
        """Test the local_training method without differential privacy."""
        federated_node = TestFederatedNode.init_test()
        federated_model = MagicMock()
        federated_model.train.return_value = (0.5, 0.8)
        result = federated_node.local_training(False, federated_model)
        assert result == {"loss": 0.5, "accuracy": 0.8, "epsilon": None}

    @staticmethod
    def test_local_training_dp() -> None:
        """Test the local_training method with differential privacy."""
        federated_node = TestFederatedNode.init_test()
        federated_model = MagicMock()
        federated_model.train_with_differential_privacy.return_value = (0.3, 0.9, 0.1)
        result = federated_node.local_training(True, federated_model)
        assert result == {"loss": 0.3, "accuracy": 0.9, "epsilon": 0.1}

    @staticmethod
    def test_send_and_receive_weights() -> None:
        """Test the send_and_receive_weights method."""
        federated_node = TestFederatedNode.init_test()
        federated_model = MagicMock()
        federated_model.get_weights.return_value = torch.Tensor([2, 3, 4])
        federated_node.receive_data_from_server = MagicMock()
        federated_node.receive_data_from_server.return_value = Weights(
            weights=torch.Tensor([2, 3, 4]),
            sender="server",
            epsilon=0.1,
        )
        metrics = {"epsilon": 0.1}
        server_channel = CommunicationChannel()
        federated_node.add_server_channel(server_channel)
        # call the function
        result = federated_node.send_and_receive_weights_with_server(
            federated_model, metrics
        )
        # assert that the returned object is of the Weights class
        assert isinstance(result, Weights)
        # assert that the weights sent to the server are the same as the weights returned by get_weights()
        assert torch.all(torch.eq(result.weights, federated_model.get_weights()))
        # assert that the sender of the weights is the node_id
        assert result.sender == "server"
        # assert that the epsilon of the weights is the same as in the metrics
        assert result.epsilon == metrics["epsilon"]
        federated_node.receive_data_from_server.assert_called_once()
        assert federated_node.message_counter == 2

    @staticmethod
    def test_start_server_phase() -> None:
        """Test the start_server_phase method."""

    @staticmethod
    def test_send_performances() -> None:
        """Test the send_performances method."""
        federated_node = TestFederatedNode.init_test()
        performances = Performances(
            node_name="node1",
            epochs=10,
            loss_list=[0.1, 0.2, 0.3, 0.4],
            accuracy_list=[0.9, 0.8, 0.7, 0.6],
            loss=0.5,
            accuracy=0.5,
            message_counter=10,
            epsilon_list=[1, 2, 3, 4],
        )
        server_channel = CommunicationChannel()
        federated_node.add_server_channel(server_channel)
        federated_node.send_performances(performances)
        received_data = server_channel.receive_data()
        assert received_data.accuracy == performances.accuracy
        assert received_data.accuracy_list == performances.accuracy_list
        assert received_data.epochs == performances.epochs
        assert received_data.epsilon_list == performances.epsilon_list
        assert received_data.loss == performances.loss
        assert received_data.loss_list == performances.loss_list
        assert received_data.message_counter == performances.message_counter
        assert received_data.node_name == performances.node_name

    @staticmethod
    def test_send_performances_should_raise_an_exception() -> None:
        """Test the send_performances method that should raise an exception."""
        federated_node = TestFederatedNode.init_test()
        performances = Performances(
            node_name="node1",
            epochs=10,
            loss_list=[0.1, 0.2, 0.3, 0.4],
            accuracy_list=[0.9, 0.8, 0.7, 0.6],
            loss=0.5,
            accuracy=0.5,
            message_counter=10,
            epsilon_list=[1, 2, 3, 4],
        )
        with pytest.raises(NotYetInitializedServerChannelError):
            federated_node.send_performances(performances)

    @staticmethod
    def test_compute_performances() -> None:
        """Test the compute_performances method."""
        federated_node = TestFederatedNode.init_test()
        performances = federated_node.compute_performances(
            loss_list=[0.1, 0.2, 0.3, 0.4],
            accuracy_list=[0.9, 0.8, 0.7, 0.6],
            phase="p2p",
            message_counter=10,
            epsilon_list=[1, 2, 3, 4],
        )
        assert performances["p2p"].loss_list == [0.1, 0.2, 0.3, 0.4]
        assert performances["p2p"].accuracy_list == [0.9, 0.8, 0.7, 0.6]
        assert performances["p2p"].message_counter == 10
        assert performances["p2p"].epsilon_list == [1, 2, 3, 4]
        assert performances["p2p"].node_name == "0_cluster_0"

    @staticmethod
    def test_receive_starting_model_from_server() -> None:
        """Test the receive_starting_model_from_server method."""
        model = nn.Linear(10, 10)
        weights = model.state_dict()

        federated_node = TestFederatedNode.init_test()
        federated_model = federated_node.init_federated_model(model)
        weights_ = Weights(weights, "server")
        federated_node.receiver_channel.send_data(weights_)

        federated_node.receive_starting_model_from_server(federated_model)
        assert federated_model.get_weights(), weights
