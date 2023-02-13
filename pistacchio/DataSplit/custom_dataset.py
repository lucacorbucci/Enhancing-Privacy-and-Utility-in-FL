import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    """Definition of generic custom dataset."""

    def __init__(self, samples: list, targets: list) -> None:
        """Initialization of the dataset.

        Args:
            samples (list): list of samples
            targets (list): list of targets
        """
        self.data: list = samples
        self.targets = np.asarray(targets)
        self.indices = np.asarray(range(len(self.targets)))

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
        return self.data[idx], self.targets[idx]


class CelebaGenderDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms = None,
    ) -> None:
        """Initialization of the dataset.

        Args:
            csv_path (str): path of the csv file with all the information
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
        """
        dataframe = pd.read_csv(csv_path)
        self.targets = dataframe["Target"]
        self.classes = dataframe["Target"]
        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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

        return transforms.functional.to_tensor(img), self.targets[index]

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples
