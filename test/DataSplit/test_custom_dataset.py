import os
import sys
from typing import Any, Tuple

import pytest
import torch
from torchvision import transforms


sys.path.insert(1, os.path.join(sys.path[0], "../.."))

from pistacchio_simulator.DataSplit.custom_dataset import CelebaGenderDataset, MyDataset


class TestMyDataset:
    """_summary_."""

    @staticmethod
    @pytest.mark.parametrize(
        ("samples", "targets", "idx", "expected_output"),
        [
            # test for when idx is 0
            ([1, 2, 3], [4, 5, 6], 0, (1, 4)),
            # test for when idx is 1
            ([1, 2, 3], [4, 5, 6], 1, (2, 5)),
            # test for when idx is 2
            ([1, 2, 3], [4, 5, 6], 2, (3, 6)),
        ],
    )
    def test_my_dataset(samples: list, targets: list, idx: int, expected_output: Any):
        """_summary_.

        Args:
            samples (list): _description_
            targets (list): _description_
            idx (int): _description_
            expected_output (Any): _description_
        """
        dataset = MyDataset(samples, targets)
        assert dataset[idx] == expected_output


class TestCelebGenderDataset:
    """_summary_."""

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize((64, 64)),
        ],
    )

    @staticmethod
    @pytest.mark.parametrize(
        ("csv_path", "image_path", "transform", "idx", "expected_output"),
        [
            # test for when idx is 0
            (
                "./test/files/celeba/celeba.csv",
                "./test/files/celeba/",
                transform,
                0,
                (torch.Tensor, 2),
            ),
            # test for when idx is 1
            (
                "./test/files/celeba/celeba.csv",
                "./test/files/celeba/",
                transform,
                1,
                (torch.Tensor, 2),
            ),
            # test for when idx is 2
            (
                "./test/files/celeba/celeba.csv",
                "./test/files/celeba",
                transform,
                2,
                (torch.Tensor, 2),
            ),
        ],
    )
    def test_celeb_gender_dataset(
        csv_path: str,
        image_path: str,
        transform,
        idx: int,
        expected_output: Tuple[torch.Tensor, torch.Tensor],
    ):
        """_summary_.

        Args:
            csv_path (str): _description_
            image_path (str): _description_
            transform (_type_): _description_
            idx (int): _description_
            expected_output (Tuple[torch.Tensor, torch.Tensor]): _description_
        """
        dataset = CelebaGenderDataset(csv_path, image_path, transform)
        assert dataset[idx][1] == expected_output[1]
        assert isinstance(dataset[idx][0], expected_output[0])

    @staticmethod
    @pytest.mark.parametrize(
        ("csv_path", "image_path", "transform"),
        [
            # test for when idx is 0
            (
                "./test/files/celeba/celeba.csv",
                "./test/files/celeba/",
                transform,
            ),
        ],
    )
    def test_celeb_gender_dataset_len(csv_path: str, image_path: str, transform):
        """_summary_.

        Args:
            csv_path (str): _description_
            image_path (str): _description_
            transform (_type_): _description_
        """
        dataset = CelebaGenderDataset(csv_path, image_path, transform)
        assert len(dataset) == len(dataset.samples)
