import os
import sys
from collections import OrderedDict

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.utils import Utils

preferences = Preferences(
    dataset_name="mnist",
    task="federatedlearning",
    mode="p2p",
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


class TestUtils:
    """Test the Utils class."""

    @staticmethod
    def test_compute_average() -> None:
        """This function is used to test the average computation."""
        shared_data = {}
        shared_data["Node1"] = {
            "test": torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]),
            "test2": torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0]),
        }
        shared_data["Node2"] = {
            "test": torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0]),
            "test2": torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0]),
        }
        shared_data["Node3"] = {
            "test": torch.tensor([9.0, 9.0, 9.0, 9.0, 9.0]),
            "test2": torch.tensor([13.0, 13.0, 13.0, 13.0, 13.0]),
        }

        expected = OrderedDict(
            [
                ("test", torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])),
                ("test2", torch.tensor([7.0, 7.0, 7.0, 7.0, 7.0])),
            ],
        )
        names = ["test", "test2"]
        result = Utils.compute_average(shared_data)

        for name in names:
            assert torch.all(torch.eq(result[name], expected[name]))

    @staticmethod
    def test_compute_distance_from_mean():
        """Test that the function computes the correct distance from mean for each node and layer."""
        shared_data = {
            "node1": {
                "layer1": torch.tensor([1, 2, 3]),
                "layer2": torch.tensor([4, 5, 6]),
            },
            "node2": {
                "layer1": torch.tensor([10, 20, 30]),
                "layer2": torch.tensor([40, 50, 60]),
            },
        }
        average_weights = {
            "layer1": torch.tensor([5.5, 11, 16.5]),
            "layer2": torch.tensor([22, 27.5, 33]),
        }

        distances = Utils.compute_distance_from_mean(shared_data, average_weights)

        assert distances == {
            "node1": torch.tensor(-15.75),
            "node2": torch.tensor(15.75),
        }

    @staticmethod
    def test_compute_distance_from_mean_with_empty_shared_data():
        """Test that the function handle empty shared data correctly."""
        shared_data = {}
        average_weights = {
            "layer1": torch.tensor([5.5, 11, 16.5]),
            "layer2": torch.tensor([22, 27.5, 33]),
        }

        distances = Utils.compute_distance_from_mean(shared_data, average_weights)

        assert distances == {}

    @staticmethod
    def test_compute_distance_from_mean_with_missing_average_weights():
        """Test that the function handle missing average weights correctly."""
        shared_data = {
            "node1": {
                "layer1": torch.tensor([1, 2, 3]),
                "layer2": torch.tensor([4, 5, 6]),
            },
            "node2": {
                "layer1": torch.tensor([10, 20, 30]),
                "layer2": torch.tensor([40, 50, 60]),
            },
        }
        average_weights = {"layer1": torch.tensor([5.5, 11, 16.5])}

        with pytest.raises(KeyError):
            Utils.compute_distance_from_mean(shared_data, average_weights)

    @staticmethod
    def test_get_run_name() -> None:
        """This function is used to test the run name generation."""
        assert (
            Utils.get_run_name(preferences)[:-8]
            == "mnist_5_nodes_1_clusters_noise_multiplier_[0.5]_max_grad_norm_[10.0]_"
        )

    @staticmethod
    def test_change_weight_names():
        """Test the change_weight_names function."""
        weights = OrderedDict({"conv1": 1, "conv2": 2, "fc1": 3})
        string_to_add = "test_"
        expected_output = OrderedDict({"test_conv1": 1, "test_conv2": 2, "test_fc1": 3})
        assert Utils.change_weight_names(weights, string_to_add) == expected_output

    @staticmethod
    def test_change_weight_names_empty_input() -> None:
        """Test the function change_weight_names with an empty input."""
        weights = OrderedDict()
        string_to_add = "test_"
        expected_output = OrderedDict()
        assert Utils.change_weight_names(weights, string_to_add) == expected_output

    @staticmethod
    def test_change_weight_names_empty_string() -> None:
        """Test the function change_weight_names with an empty string to add."""
        weights = OrderedDict({"conv1": 1, "conv2": 2, "fc1": 3})
        string_to_add = ""
        expected_output = OrderedDict({"conv1": 1, "conv2": 2, "fc1": 3})
        assert Utils.change_weight_names(weights, string_to_add) == expected_output

    @staticmethod
    def test_shuffle_list() -> None:
        """Test the shuffle_lists function."""
        first_list = [1, 2, 3, 4]
        second_list = ["a", "b", "c", "d"]
        first_list_shuffled, second_list_shuffled = Utils.shuffle_lists(
            first_list, second_list
        )
        assert len(first_list_shuffled) == len(first_list)
        assert len(second_list_shuffled) == len(second_list)
        assert set(first_list_shuffled) == set(first_list)
        assert set(second_list_shuffled) == set(second_list)

    @staticmethod
    def test_shuffle_list_with_empty_input() -> None:
        """Test the shuffle_lists function with empty input."""
        first_list = []
        second_list = []
        with pytest.raises(ValueError):
            Utils.shuffle_lists(first_list, second_list)

    @staticmethod
    def test_shuffle_list_with_different_length() -> None:
        """Test the shuffle_lists function with different length lists."""
        first_list = [1, 2, 3]
        second_list = ["a", "b", "c", "d"]
        with pytest.raises(ValueError):
            Utils.shuffle_lists(first_list, second_list)
