import os
import sys
from typing import Mapping, TypeVar

import numpy as np
import pytest
import torch
from opacus.grad_sample.grad_sample_module import GradSampleModule
from torch import Tensor, nn, optim

TDestination = TypeVar("TDestination", bound=Mapping[str, Tensor])

sys.path.insert(1, os.path.join(sys.path[0], "../.."))

from pistacchio_simulator.DataSplit.custom_dataset import MyDataset
from pistacchio_simulator.DataSplit.data_split import DataSplit
from pistacchio_simulator.DataSplit.storage_manager import StorageManager
from pistacchio_simulator.Models.federated_model import FederatedModel
from pistacchio_simulator.Models.mnist import MnistNet
from pistacchio_simulator.Utils.data_loader import DataLoader
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.task import Task

preferences_mnist = Preferences(
    dataset_name="mnist",
    mode="classic",
    task="federatedlearning",
    debug=False,
    save_model=False,
    wandb=False,
    data_split_config={
        "split_type": "stratified",
        "num_nodes": 8,
        "num_clusters": 8,
        "server_test_set": "server_validation_split",
    },
    p2p_config={
        "local_training_epochs": 1,
        "diff_privacy_p2p": False,
        "num_communication_round_pre_training": [0],
    },
    server_config={
        "diff_privacy_server": False,
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

preferences_cifar = Preferences(
    dataset_name="cifar10",
    mode="classic",
    task="federatedlearning",
    debug=False,
    save_model=False,
    wandb=False,
    data_split_config={
        "split_type": "stratified",
        "num_nodes": 8,
        "num_clusters": 8,
        "server_test_set": "server_validation_split",
    },
    p2p_config={
        "local_training_epochs": 1,
        "diff_privacy_p2p": False,
        "num_communication_round_pre_training": 1,
    },
    server_config={
        "diff_privacy_server": False,
        "num_communication_round_with_server": 1,
    },
    hyperparameters={
        "batch_size": 32,
        "lr": 0.0001,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "DELTA": 1e-5,
        "noise_multiplier": 2.0,
        "noise_multiplier_P2P": 0.5,
        "max_grad_norm": 10.0,
        "weight_decay": 0,
        "min_improvement": 0.001,
        "min_accuracy": 0.9,
        "patience": 1,
    },
)

preferences_cifar_epsilon = Preferences(
    dataset_name="cifar10",
    mode="classic",
    task="federatedlearning",
    debug=False,
    save_model=False,
    wandb=False,
    data_split_config={
        "split_type": "stratified",
        "num_nodes": 8,
        "num_clusters": 8,
        "server_test_set": "server_validation_split",
    },
    p2p_config={
        "local_training_epochs": 1,
        "diff_privacy_p2p": False,
        "num_communication_round_pre_training": 1,
    },
    server_config={
        "diff_privacy_server": False,
        "num_communication_round_with_server": 1,
    },
    hyperparameters={
        "batch_size": 32,
        "lr": 0.0001,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "DELTA": 1e-5,
        "EPSILON": 1.0,
        "EPSILON_P2P": 1.0,
        "max_grad_norm": 10.0,
        "weight_decay": 0,
        "min_improvement": 0.001,
        "min_accuracy": 0.9,
        "patience": 1,
    },
)


class MyNetwork(nn.Module):
    """Definition of a simple mock neural network
    with input and output layer
    that will be used for testing purposes.
    """

    def __init__(self, input_size, output_size) -> None:
        """Initializes the network.

        Args:
            input_size (_type_): size of the input layer
            output_size (_type_): size of the output layer
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (_type_): input

        Returns
        -------
            _type_: the computed output
        """
        x = self.fc1(x)
        return x


class MockNetwork(torch.nn.Module):
    """Definition of a simple neural network
    with 3 layers that can be used for testing
    purposes.
    """

    def __init__(self, input_size, hidden_size, output_size) -> None:
        """Initializes the network.

        Args:
            input_size (_type_): size of the input layer
            output_size (_type_): size of the output layer
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (_type_): input

        Returns
        -------
            _type_: the computed output
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def generate_mock_weights(input_size, hidden_size, output_size) -> dict:
    """Generates mock weights for a simple network.

    Args:
        input_size (_type_): size of the input layer
        hidden_size (_type_): size of the hidden layer
        output_size (_type_): size of the output layer

    Returns
    -------
        _type_: the generated weights
    """
    # Initialize the weights randomly
    mock_fc1_weights = torch.randn(hidden_size, input_size)
    mock_fc1_bias = torch.randn(hidden_size)
    mock_fc2_weights = torch.randn(output_size, hidden_size)
    mock_fc2_bias = torch.randn(output_size)
    # Pack the weights and biases into a dictionary
    mock_weights = {
        "fc1.weight": mock_fc1_weights,
        "fc1.bias": mock_fc1_bias,
        "fc2.weight": mock_fc2_weights,
        "fc2.bias": mock_fc2_bias,
    }

    return mock_weights


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

    # Generate 1000 random samples of size 10 with 5 classes with torch rand
    X = torch.rand(100, 10)
    # Generate 1000 random targets
    Y = torch.randint(0, 5, (100,))

    # Initialize the dataset and the model
    my_dataset = MyDataset(X, Y)
    my_dataset.classes = Y
    my_dataset.targets = Y

    # Define the percentage configuration
    percentage_configuration = {
        "cluster_0": {0: 100, 1: 100, 2: 100},
        "cluster_1": {3: 100, 4: 100},
    }

    cluster_datasets, _ = DataSplit.percentage_split(
        dataset=my_dataset,
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
        dataset_name="cifar10",
        splitted_dataset=cluster_datasets,
        dataset_type="train_set",
        names=names,
    )
    StorageManager.write_splitted_dataset(
        dataset_name="cifar10",
        splitted_dataset=cluster_datasets,
        dataset_type="test_set",
        names=names,
    )


class TestFederatedModel:
    """_summary_."""

    @staticmethod
    def test_create_federated_model() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        training_set = DataLoader().load_splitted_dataset_train(
            "../data/mnist/federated_split/train_set/0_cluster_0_split",
        )
        assert federated_model.dataset_name == "mnist"
        assert federated_model.node_name == "0_cluster_0"
        assert federated_model.preferences == preferences_mnist
        assert len(federated_model.training_set) == len(training_set)

    @staticmethod
    def test_init_model() -> None:
        """Test the initialization of the model."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        federated_model.init_model(MnistNet())

        assert federated_model.optimizer is not None
        assert federated_model.trainloader is not None
        assert federated_model.testloader is not None
        assert federated_model.net is not None

    @staticmethod
    def test_add_model() -> None:
        """Test the addition of a model to the fedeated model."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        federated_model.add_model(MnistNet())

        assert federated_model.net is not None

    @staticmethod
    def test_load_data() -> None:
        """Test the loading of the data."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        training_set = DataLoader().load_splitted_dataset_train(
            "../data/mnist/federated_split/train_set/0_cluster_0_split",
        )
        test_set = DataLoader().load_splitted_dataset_train(
            "../data/mnist/federated_split/test_set/0_cluster_0_split",
        )
        trainloader, testloader = federated_model.load_data()

        assert len(training_set) == len(trainloader.dataset)
        assert len(test_set) == len(testloader.dataset)

    @staticmethod
    def test_get_weights_list() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )

        federated_model.init_model(MnistNet())
        weights_list = federated_model.get_weights_list()
        assert len(weights_list) == 6

    @staticmethod
    def test_get_weights_list_should_raise_exception() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        with pytest.raises(Exception):
            federated_model.get_weights_list()

    @staticmethod
    def test_get_weights() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        federated_model.init_model(MnistNet())
        weights = federated_model.get_weights()
        assert len(weights) == 6

    @staticmethod
    def test_get_weights_should_raise_exception() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        with pytest.raises(Exception):
            federated_model.get_weights()

    @staticmethod
    def test_update_weights() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        federated_model.init_model(
            MockNetwork(input_size=10, hidden_size=20, output_size=1)
        )
        weights = generate_mock_weights(input_size=10, hidden_size=20, output_size=1)
        federated_model.update_weights(weights)
        updated_weights = federated_model.get_weights()
        for key, _ in weights.items():
            assert np.array_equal(weights[key], updated_weights[key])

    @staticmethod
    def test_update_weights_should_raise_an_exception() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )
        weights = generate_mock_weights(input_size=10, hidden_size=20, output_size=1)
        with pytest.raises(Exception):
            federated_model.update_weights(weights)

    @staticmethod
    def test_train_should_raise_exception() -> None:
        """_summary_."""
        # Initialize FederatedLearning object with mock_dataset and mock_network
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )

        with pytest.raises(Exception):
            federated_model.train()

    @staticmethod
    def test_train() -> None:
        """_summary_."""
        my_network = MyNetwork(input_size=10, output_size=5)

        federated_model = FederatedModel(
            dataset_name=preferences_cifar.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_cifar,
        )
        federated_model.init_model(my_network)
        # Call train() and store the return values
        loss, accuracy = federated_model.train()

        # Assert that loss and accuracy are of correct type
        assert isinstance(loss, float)
        assert isinstance(accuracy, Tensor)

        # Assert that loss and accuracy are within a reasonable range
        assert loss >= 0
        assert 0 <= accuracy <= 1

    @staticmethod
    def test_train_with_differential_privacy_should_raise_exception() -> None:
        """_summary_."""
        # Initialize FederatedLearning object with mock_dataset and mock_network
        federated_model = FederatedModel(
            dataset_name=preferences_mnist.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_mnist,
        )

        with pytest.raises(Exception):
            federated_model.train_with_differential_privacy()

    @staticmethod
    def test_init_differential_privacy() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_cifar.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_cifar,
        )
        my_network = MyNetwork(input_size=10, output_size=5)
        federated_model.init_model(my_network)
        federated_model.init_differential_privacy(phase=Phase.SERVER)

        model, optimizer, trainloader = federated_model.init_differential_privacy(
            Phase.SERVER
        )
        # Assert that model is of correct type
        assert isinstance(model, GradSampleModule)
        assert isinstance(model._module, MyNetwork)

        # Assert that optimizer is of correct type
        assert isinstance(optimizer, optim.Optimizer)

        # Assert that trainloader is of correct type
        assert isinstance(trainloader, torch.utils.data.DataLoader)

        # Call init_differential_privacy() with Phase.CLIENT and store the return values
        model, optimizer, trainloader = federated_model.init_differential_privacy(
            Phase.MIXED
        )

        # Assert that model is of correct type
        assert isinstance(model, GradSampleModule)
        assert isinstance(model._module, MyNetwork)

        # Assert that optimizer is of correct type
        assert isinstance(optimizer, optim.Optimizer)

        # Assert that trainloader is of correct type
        assert isinstance(trainloader, torch.utils.data.DataLoader)

        # Call init_differential_privacy() with Phase.CLIENT and store the return values
        model, optimizer, trainloader = federated_model.init_differential_privacy(
            Phase.MIXED
        )

        # Assert that model is of correct type
        assert isinstance(model, GradSampleModule)
        assert isinstance(model._module, MyNetwork)

        # Assert that optimizer is of correct type
        assert isinstance(optimizer, optim.Optimizer)

        # Assert that trainloader is of correct type
        assert isinstance(trainloader, torch.utils.data.DataLoader)

        assert federated_model.diff_privacy_initialized is True

    @staticmethod
    def test_init_differential_privacy_with_epsilon() -> None:
        """_summary_."""
        federated_model = FederatedModel(
            dataset_name=preferences_cifar.dataset_name,
            node_name="0_cluster_0",
            preferences=preferences_cifar_epsilon,
        )
        my_network = MyNetwork(input_size=10, output_size=5)
        federated_model.init_model(my_network)
        federated_model.init_differential_privacy(phase=Phase.SERVER)

        model, optimizer, trainloader = federated_model.init_differential_privacy(
            Phase.SERVER
        )
        # Assert that model is of correct type
        assert isinstance(model, GradSampleModule)
        assert isinstance(model._module, MyNetwork)

        # Assert that optimizer is of correct type
        assert isinstance(optimizer, optim.Optimizer)

        # Assert that trainloader is of correct type
        assert isinstance(trainloader, torch.utils.data.DataLoader)

        # Call init_differential_privacy() with Phase.CLIENT and store the return values
        model, optimizer, trainloader = federated_model.init_differential_privacy(
            Phase.MIXED
        )

        # Assert that model is of correct type
        assert isinstance(model, GradSampleModule)
        assert isinstance(model._module, MyNetwork)

        # Assert that optimizer is of correct type
        assert isinstance(optimizer, optim.Optimizer)

        # Assert that trainloader is of correct type
        assert isinstance(trainloader, torch.utils.data.DataLoader)

        # Call init_differential_privacy() with Phase.CLIENT and store the return values
        model, optimizer, trainloader = federated_model.init_differential_privacy(
            Phase.MIXED
        )

        # Assert that model is of correct type
        assert isinstance(model, GradSampleModule)
        assert isinstance(model._module, MyNetwork)

        # Assert that optimizer is of correct type
        assert isinstance(optimizer, optim.Optimizer)

        # Assert that trainloader is of correct type
        assert isinstance(trainloader, torch.utils.data.DataLoader)

        assert federated_model.diff_privacy_initialized is True
