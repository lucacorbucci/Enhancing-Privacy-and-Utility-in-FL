from typing import Any

import dill
import torch
from torch.utils.data import Dataset


class DataLoader:
    """This class is used to load the Mnist dataset."""

    @staticmethod
    def load_splitted_dataset(
        path_train: str,
    ) -> torch.utils.data.DataLoader[Any]:
        """This function loads the splitted dataset.

        Args:
            ID (int): node id

        Returns
        -------
            _type_: _description_
        """

        print(f"LOADING {path_train}")
        data = torch.load(path_train)
        return data
