import os
import sys

import pytest
import torch


sys.path.insert(1, os.path.join(sys.path[0], "../.."))

from pistacchio.DataSplit.custom_dataset import MyDataset
from pistacchio.DataSplit.data_split import DataSplit
from pistacchio.DataSplit.storage_manager import StorageManager
from pistacchio.Utils.data_loader import DataLoader
from pistacchio.Utils.task import Task


@pytest.fixture(scope="session", autouse=True)
def pytest_configure() -> None:
    """_summary_.

    Args:
        config (_type_): _description_
    """
    X = torch.rand(100, 10)
    Y = torch.tensor([0, 1, 2, 3] * 25)

    my_dataset = MyDataset(X, Y)
    my_dataset.classes = Y
    my_dataset.targets = Y

    percentage_configuration = {
        "cluster_0": {0: 100, 1: 100},
        "cluster_1": {2: 100, 3: 100},
    }

    cluster_datasets, _ = DataSplit.percentage_split(
        dataset=my_dataset,
        percentage_configuration=percentage_configuration,
        num_workers=2,
        task=Task("federatedlearning"),
    )

    names = [
        f"{node_id}_cluster_{cluster_id}"
        for cluster_id in range(2)
        for node_id in range(2)
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


class TestDataLoader:
    @staticmethod
    def test_load_splitted_dataset_train() -> None:
        """Test the load_splitted_dataset_train function."""
        training_set = DataLoader().load_splitted_dataset_train(
            "../data/cifar10/federated_split/train_set/0_cluster_0_split",
        )
        assert len(training_set) == 25

    @staticmethod
    def test_load_splitted_dataset_test() -> None:
        """Test the test_load_splitted_dataset_test function."""
        test_set = DataLoader().load_splitted_dataset_test(
            "../data/cifar10/federated_split/test_set/0_cluster_0_split",
        )
        assert len(test_set) == 25
