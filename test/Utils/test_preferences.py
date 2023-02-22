import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.task import TaskType

preferences_mnist = {
    "dataset": "mnist",
    "mode": "semi_p2p",
    "task": "federatedlearning",
    "debug": False,
    "save_model": False,
    "federated": True,
    "wandb": False,
    "data_split_config": {
        "split_type": "percentage",
        "num_classes": 10,
        "num_nodes": 1,
        "num_clusters": 2,
        "num_classes_per_cluster": 2,
        "server_validation_set": "server_validation_split",
        "percentage_configuration": {
            "cluster_1": {"1": 70, "2": 40, "3": 20, "4": 20},
            "cluster_0": {"0": 60, "1": 30, "2": 20, "3": 20},
            "cluster_2": {"2": 40, "3": 20, "4": 20, "5": 20},
            "cluster_3": {"3": 40, "4": 20, "5": 20, "6": 30},
            "cluster_4": {"4": 40, "5": 20, "6": 30, "7": 10},
            "cluster_5": {"5": 40, "6": 20, "7": 30, "8": 30},
            "cluster_6": {"6": 20, "7": 40, "8": 30, "9": 70},
            "cluster_7": {"7": 20, "8": 40, "9": 30, "0": 40},
        },
    },
    "p2p_config": {
        "differential_privacy_pre_training_cluster": False,
        "differential_privacy_mixed_mode_cluster": False,
        "local_training_epochs_in_cluster": 1,
        "num_communication_round_pre_training": [1],
        "num_communication_round_mixed_mode": 2,
    },
    "server_config": {
        "differential_privacy_server": True,
        "mixed_mode": False,
        "local_training_epochs_with_server": 1,
        "num_communication_round_with_server": 1,
        "total_mixed_iterations": 1,
    },
    "hyperparameters": {
        "batch_size": 32,
        "lr": 0.001,
        "MAX_PHYSICAL_BATCH_SIZE": 128,
        "DELTA": 1e-5,
        "noise_multiplier": [2.0],
        "noise_multiplier_P2P": [0.5],
        "max_grad_norm": [1.2],
        "weight_decay": 0,
        "min_improvement": 0.001,
        "patience": 10,
        "min_accuracy": 0.80,
    },
}


class TestPreferences:
    """_summary_."""

    @staticmethod
    def test_initialization() -> None:
        """_summary_."""
        preferences = Preferences(
            task="federatedlearning",
            dataset_name="mnist",
            mode="classic",
            debug=True,
            save_model=True,
            hyperparameters={
                "batch_size": 32,
            },
            data_split_config={
                "split_type": "stratified",
            },
            p2p_config={
                "local_training_epochs": 1,
            },
            server_config={
                "diff_privacy_server": False,
            },
            wandb=True,
        )
        assert preferences.dataset_name == "mnist"
        assert preferences.task.task_type == TaskType.FEDERATEDLEARNING
        assert preferences.mode == "classic"
        assert preferences.wandb
        assert preferences.debug
        assert preferences.save_model
        assert preferences.hyperparameters == {
            "batch_size": 32,
        }
        assert preferences.data_split_config == {
            "split_type": "stratified",
        }
        assert preferences.p2p_config == {
            "local_training_epochs": 1,
        }
        assert preferences.server_config == {
            "diff_privacy_server": False,
        }

    @staticmethod
    def test_generate_from_json() -> None:
        """_summary_."""
        preferences = Preferences.generate_from_json(preferences_mnist)
        assert preferences.dataset_name == "mnist"
        assert preferences.task.task_type == TaskType.FEDERATEDLEARNING
        assert preferences.mode == "semi_p2p"
        assert not preferences.wandb
        assert not preferences.debug
        assert not preferences.save_model
        assert preferences.hyperparameters == {
            "batch_size": 32,
            "lr": 0.001,
            "MAX_PHYSICAL_BATCH_SIZE": 128,
            "DELTA": 1e-5,
            "noise_multiplier": [2.0],
            "noise_multiplier_P2P": [0.5],
            "max_grad_norm": [1.2],
            "weight_decay": 0,
            "min_improvement": 0.001,
            "patience": 10,
            "min_accuracy": 0.80,
        }
        assert preferences.data_split_config == {
            "split_type": "percentage",
            "num_classes": 10,
            "num_nodes": 1,
            "num_clusters": 2,
            "num_classes_per_cluster": 2,
            "server_validation_set": "server_validation_split",
            "percentage_configuration": {
                "cluster_1": {"1": 70, "2": 40, "3": 20, "4": 20},
                "cluster_0": {"0": 60, "1": 30, "2": 20, "3": 20},
                "cluster_2": {"2": 40, "3": 20, "4": 20, "5": 20},
                "cluster_3": {"3": 40, "4": 20, "5": 20, "6": 30},
                "cluster_4": {"4": 40, "5": 20, "6": 30, "7": 10},
                "cluster_5": {"5": 40, "6": 20, "7": 30, "8": 30},
                "cluster_6": {"6": 20, "7": 40, "8": 30, "9": 70},
                "cluster_7": {"7": 20, "8": 40, "9": 30, "0": 40},
            },
        }
        assert preferences.p2p_config == {
            "differential_privacy_pre_training_cluster": False,
            "differential_privacy_mixed_mode_cluster": False,
            "local_training_epochs_in_cluster": 1,
            "num_communication_round_pre_training": [1],
            "num_communication_round_mixed_mode": 2,
        }

        assert preferences.server_config == {
            "differential_privacy_server": True,
            "mixed_mode": False,
            "local_training_epochs_with_server": 1,
            "num_communication_round_with_server": 1,
            "total_mixed_iterations": 1,
        }
