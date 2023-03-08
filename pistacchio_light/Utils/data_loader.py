from typing import Any

import dill
import torch


class DataLoader:
    """This class is used to load the Mnist dataset."""

    @staticmethod
    def load_splitted_dataset_train(
        path_train: str,
    ) -> torch.utils.data.DataLoader[Any]:
        """This function loads the splitted train dataset.

        Args:
            ID (int): node id

        Returns
        -------
            _type_: _description_
        """
        with open(path_train, "rb") as file:
            data = dill.load(file)
        return data

    @staticmethod
    def load_splitted_dataset_test(
        path_test: str,
    ) -> torch.utils.data.DataLoader["Any"]:
        """This function loads the splitted test dataset.

        Args:
            ID (int): node id

        Returns
        -------
            _type_: _description_
        """
        with open(path_test, "rb") as file:
            data = dill.load(file)
        return data
