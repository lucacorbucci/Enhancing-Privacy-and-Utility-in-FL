import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TabularDataset(Dataset):
    def __init__(self, x, z, y):
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.
        """
        self.samples = x
        self.sensitive_features = z
        self.sensitive_attribute = z
        self.gender = z
        self.classes = y
        self.targets = y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.
        """
        x_sample = self.samples[idx]
        z_sample = self.sensitive_features[idx]
        y_sample = self.targets[idx]

        return x_sample, y_sample



class TabularDataset(Dataset):
    def __init__(self, x, z, y, transform = None):
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.
        """
        self.data = x
        self.samples = x
        self.targets = y
        self.indexes = range(len(self.data))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.
        """
        x_sample = self.samples[idx]
        y_sample = self.targets[idx]

        return x_sample, y_sample

class MyDataset(Dataset):
    """Definition of generic custom dataset."""

    def __init__(self, samples: list, targets: list, transform) -> None:
        self.data: list = samples
        self.targets = torch.tensor(targets)
        self.indices = np.asarray(range(len(self.targets)))
        self.transform = transform

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            _type_: size of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve
        """
        img, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class MyDatasetWithCSV(Dataset):
    """Definition of generic custom dataset."""

    def __init__(
        self,
        targets: list,
        image_path: str,
        image_ids: list,
        sensitive_features: list,
        transform: torchvision.transforms = None,
    ) -> None:
        self.data: list = image_ids
        self.targets = torch.tensor(list(targets))
        self.classes = torch.tensor(list(targets))
        self.indices = np.asarray(range(len(self.targets)))
        self.samples = image_ids
        self.n_samples = len(self.samples)
        self.transform = transform
        self.image_path = image_path
        self.sensitive_features = sensitive_features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """
        img = Image.open(
            os.path.join(self.image_path, self.samples[index]),
        ).convert(
            "RGB",
        )

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.targets[index],
        )
        # img = Image.open(os.path.join(self.image_path, self.samples[index])).convert(
        #     "RGB",
        # )
        # # if self.transform:
        # #     img = self.transform(img)
        # if self.transform:
        #     img = self.transform(img)

        # return img, self.targets[index]

        # # if isinstance(img, torch.Tensor):
        # #     return img, self.targets[index]
        # # return transforms.functional.to_tensor(img), self.targets[index]

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples


class ImaginetteDataset(Dataset):
    """Definition of the dataset used for the Imaginette Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms = None,
    ) -> None:
        """Initialization of the dataset.

        Args:
        ----
            csv_path (str): path of the csv file with all the information
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
        """
        dataframe = pd.read_csv(csv_path)
        self.targets = dataframe["Target"]
        self.classes = dataframe["Target"]
        self.samples = list(dataframe["Path"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """

        img = Image.open(os.path.join(self.image_path, self.samples[index])).convert(
            "RGB",
        )
        if self.transform:
            img = self.transform(img)

        return img, self.targets[index]

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples


class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms,
        debug: bool = True,
    ) -> None:
        """Initialization of the dataset.

        Args:
        ----
            csv_path (str): path of the csv file with all the information
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
        """
        dataframe = pd.read_csv(csv_path)
        smiling_dict = {-1: 0, 1: 1}
        targets = [smiling_dict[item] for item in dataframe["Smiling"].tolist()]
        self.targets = targets
        self.classes = targets

        self.samples = list(dataframe["image_id"])
        # self.data = list(dataframe["image_id"])
        
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path
        self.debug = debug
        # self.data = np.array([
        #         Image.open(os.path.join(self.image_path, sample)).convert(
        #             "RGB",
        #         )
        #         for sample in self.samples
        #     ], dtype=object)
        # if not self.debug:
        #     self.images = [
        #         Image.open(os.path.join(self.image_path, sample)).convert(
        #             "RGB",
        #         )
        #         for sample in self.samples
        #     ]

    def __getitem__(self, index: int):
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """
        # if self.debug:
        #     img = Image.open(
        #         os.path.join(self.image_path, self.samples[index]),
        #     ).convert(
        #         "RGB",
        #     )
        # else:
        img = self.data[index]

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.targets[index],
        )

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples
