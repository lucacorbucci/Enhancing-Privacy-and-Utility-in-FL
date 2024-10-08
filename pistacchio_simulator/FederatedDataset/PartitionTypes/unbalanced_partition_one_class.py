import random

import numpy as np
import torch
from pistacchio_simulator.FederatedDataset.PartitionTypes.iid_partition import (
    IIDPartition,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.non_iid_partition import (
    NonIIDPartition,
)


class UnbalancedPartitionOneClass:
    def do_partitioning(
        labels: np.ndarray,
        sensitive_features: np.ndarray,
        num_partitions: int,
        total_num_classes: int,
        alpha: int,
        ratio: float,
    ) -> list:
        """

        Returns:
            list: a list of lists of indexes
        """

        sensitive_features = [item.item() for item in sensitive_features]

        # Number of nodes that will have 3 combinations
        num_nodes_with_3_combinations = int(ratio * num_partitions)

        indexes = range(len(labels))
        current_labels = [item.item() for item in labels]

        indexes_and_labels = dict(zip(indexes, current_labels))
        indexes_and_sensitive_features = dict(zip(indexes, sensitive_features))

        print(num_nodes_with_3_combinations)

        splitted_indexes = []
        indexes = torch.tensor(indexes)
        labels = torch.tensor(labels)
        splitted_indexes += NonIIDPartition.do_partitioning_with_indexes(
            indexes=indexes, labels=labels, num_partitions=num_partitions, alpha=alpha
        )
        new_index_list = []
        for index_list in splitted_indexes:
            current_list = []
            if num_nodes_with_3_combinations > 0:
                for item in index_list:
                    if (
                        indexes_and_labels[item] == 1
                        and indexes_and_sensitive_features[item] == 1
                    ):
                        continue
                    else:
                        current_list.append(item)

                new_index_list.append(current_list)
                num_nodes_with_3_combinations -= 1
            else:
                new_index_list.append(index_list)

        # splitted_indexes = list(itertools.chain.from_iterable(zip(*splitted_indexes)))
        random.shuffle(new_index_list)
        return new_index_list
