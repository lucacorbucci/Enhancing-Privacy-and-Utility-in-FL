from collections import Counter

import numpy as np
import torch

from pistacchio_simulator.FederatedDataset.Utils.lda import create_lda_partitions


class IIDPartition:
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
    ) -> dict:
        labels = dataset.targets

        idx = torch.tensor(list(range(len(labels))))
        # shuffle the indexes
        idx = idx[torch.randperm(len(idx))]

        # split the indexes into num_partitions
        splitted_indexes = np.array_split(idx, num_partitions)
        splitted_labels = [
            np.array(labels)[index_list] for index_list in splitted_indexes
        ]

        splitted_indexes_dict = {
            f"node_{index}": item for index, item in enumerate(splitted_indexes)
        }
        splitted_labels_dict = {
            f"node_{index}": item for index, item in enumerate(splitted_labels)
        }

        data = dataset.data if hasattr(dataset, "data") else dataset.samples

        partitions_data = {
            cluster: np.array(data)[indexes.tolist()]
            for cluster, indexes in splitted_indexes_dict.items()
        }

        return splitted_indexes_dict, splitted_labels_dict, partitions_data

    def do_iid_partitioning_with_indexes(
        indexes: np.array,
        num_partitions: int,
    ) -> np.array:
        """This function splits a list of indexes in N parts.
        First of all the list is shuffled and then it is splitted in N parts.

        Args:
            indexes (np.array): the list of indexes to be splitted
            num_partitions (int): the number of partitions

        Returns:
            np.array: the list of splitted indexes
        """
        idx = indexes[torch.randperm(len(indexes))]
        return np.array_split(idx, num_partitions)
