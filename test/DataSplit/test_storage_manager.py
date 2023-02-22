import os
import sys
from typing import List

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pistacchio_simulator.DataSplit.storage_manager import StorageManager


class TestStorageManager:
    """_summary_."""

    @staticmethod
    @pytest.mark.parametrize(
        ("dataset_name", "dataset_type", "names"),
        [
            # test for write_splitted_dataset method
            ("mnist", "train", ["node_1", "node_2"]),
            ("cifar10", "test", ["node_1", "node_2"]),
            ("fashion_mnist", "validation", ["node_1", "node_2"]),
        ],
    )
    def test_write_splitted_dataset(
        dataset_name: str, dataset_type: str, names: List[str]
    ):
        """_summary_.

        Args:
            dataset_name (str): _description_
            dataset_type (str): _description_
            names (List[str]): _description_
        """
        splitted_dataset = [object(), object()]
        StorageManager.write_splitted_dataset(
            dataset_name, splitted_dataset, dataset_type, names
        )

        for file_name in names:
            file_path = f"../data/{dataset_name}/federated_split/{dataset_type}/{file_name}_split"
            assert os.path.exists(file_path)
            # delete the files after the test
            os.remove(file_path)

    @staticmethod
    @pytest.mark.parametrize(
        ("dataset_name", "dataset_type"),
        [
            # test for write_validation_dataset method
            ("mnist", "validation"),
            ("cifar10", "validation"),
            ("fashion_mnist", "validation"),
        ],
    )
    def test_write_validation_dataset(dataset_name: str, dataset_type: str):
        """_summary_.

        Args:
            dataset_name (str): _description_
            dataset_type (str): _description_
        """
        dataset = object()
        StorageManager.write_validation_dataset(dataset_name, dataset, dataset_type)
        file_path = f"../data/{dataset_name}/federated_split/{dataset_type}/{dataset_type}_split"
        assert os.path.exists(file_path)
        # delete the file after the test
        os.remove(file_path)
