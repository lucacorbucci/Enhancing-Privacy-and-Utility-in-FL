from typing import Any

import dill
import torch


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

        data = torch.load(path_train)
        return data
