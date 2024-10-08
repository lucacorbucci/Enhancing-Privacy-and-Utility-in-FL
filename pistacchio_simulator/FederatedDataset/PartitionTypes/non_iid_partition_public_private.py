from collections import Counter

import numpy as np
import torch

from pistacchio_simulator.FederatedDataset.PartitionTypes.iid_partition import (
    IIDPartition,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.non_iid_partition import (
    NonIIDPartition,
)


class NonIIDPartitionPublicPrivate:
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
        alpha=1000000,
    ) -> dict:
        if not alpha:
            raise ValueError("Alpha must be a positive number")
        labels = dataset.targets
        idx = torch.tensor(list(range(len(labels))))

        index_per_label = {}
        for index, label in zip(idx, labels):
            if label.item() not in index_per_label:
                index_per_label[label.item()] = []
            index_per_label[label.item()].append(index.item())

        # in list labels we have the labels of this dataset
        list_labels = {item.item() for item in labels}

        to_be_sampled = []
        # create the distribution for each class
        for label in list_labels:
            # For each label we want a distribution over the num_partitions
            distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
            # we have to sample from the group of samples that have label equal
            # to label and not from the entire labels list.
            selected_labels = labels[labels == label]
            tmp_to_be_sampled = np.random.choice(
                num_partitions, len(selected_labels), p=distribution[0]
            )
            # Inside to_be_sampled we save a counter for each label
            # The counter is the number of samples that we want to sample for each
            # partition
            to_be_sampled.append(Counter(tmp_to_be_sampled))
        # create the partitions
        partitions_index = {
            f"node_{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        public_private_indexes = {}

        for class_index, distribution_samples in zip(list_labels, to_be_sampled):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"node_{cluster_name}"] += index_per_label[
                    class_index
                ][:samples]

                index_per_label[class_index] = index_per_label[class_index][samples:]

        for node_name, indexes in partitions_index.items():
            public_and_private = IIDPartition.do_iid_partitioning_with_indexes(
                indexes=torch.tensor(indexes), num_partitions=2
            )
            public_private_indexes[f"{node_name}_private"] = public_and_private[0]
            public_private_indexes[f"{node_name}_public"] = public_and_private[1]

        total = 0
        for cluster, samples in public_private_indexes.items():
            total += len(samples)

        assert total == len(labels)

        partitions_labels = {
            cluster: [item.item() for item in labels[samples]]
            for cluster, samples in public_private_indexes.items()
        }

        return public_private_indexes, partitions_labels

    def do_partitioning_different_distribution(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
        alpha=1000000,
        previous_distribution_per_labels: dict = None,
    ) -> dict:
        if not alpha:
            raise ValueError("Alpha must be a positive number")
        labels = dataset.targets
        idx = torch.tensor(list(range(len(labels))))

        index_per_label = {}
        for index, label in zip(idx, labels):
            if label.item() not in index_per_label:
                index_per_label[label.item()] = []
            index_per_label[label.item()].append(index.item())

        # in list labels we have the labels of this dataset
        list_labels = {item.item() for item in labels}

        distribution_per_labels = {}
        to_be_sampled = []
        # create the distribution for each class

        for label in list_labels:
            if previous_distribution_per_labels:
                distribution = previous_distribution_per_labels[label]
            else:
                # For each label we want a distribution over the num_partitions
                distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
                distribution_per_labels[label] = distribution
            # we have to sample from the group of samples that have label equal
            # to label and not from the entire labels list.
            selected_labels = labels[labels == label]
            tmp_to_be_sampled = np.random.choice(
                num_partitions, len(selected_labels), p=distribution[0]
            )
            # Inside to_be_sampled we save a counter for each label
            # The counter is the number of samples that we want to sample for each
            # partition
            to_be_sampled.append(Counter(tmp_to_be_sampled))
        # create the partitions
        partitions_index = {
            f"node_{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        public_private_indexes = {}

        for class_index, distribution_samples in zip(list_labels, to_be_sampled):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"node_{cluster_name}"] += index_per_label[
                    class_index
                ][:samples]

                index_per_label[class_index] = index_per_label[class_index][samples:]

        for node_name, indexes in partitions_index.items():
            # public_and_private = IIDPartition.do_iid_partitioning_with_indexes(indexes=torch.tensor(indexes), num_partitions=2)
            public_and_private = NonIIDPartition.do_iid_partitioning_with_indexes(
                targets=labels[indexes],
                data=dataset.data,
                idx=torch.tensor(indexes),
                num_partitions=2,
                total_num_classes=total_num_classes,
                alpha=1.0,
            )

            public_private_indexes[f"{node_name}_private"] = public_and_private[0]
            public_private_indexes[f"{node_name}_public"] = public_and_private[1]

        total = 0
        for cluster, samples in public_private_indexes.items():
            total += len(samples)

        assert total == len(labels)

        partitions_labels = {
            cluster: [item.item() for item in labels[samples]]
            for cluster, samples in public_private_indexes.items()
        }

        return public_private_indexes, partitions_labels, distribution_per_labels

    def do_partitioning_different_distribution_nodes(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
        alpha=1000000,
        max_size: float = None,
        previous_distribution_per_labels: dict = None,
    ) -> dict:
        if not alpha:
            raise ValueError("Alpha must be a positive number")
        labels = dataset.targets
        idx = torch.tensor(list(range(len(labels))))

        index_per_label = {}
        for index, label in zip(idx, labels):
            if label.item() not in index_per_label:
                index_per_label[label.item()] = []
            index_per_label[label.item()].append(index.item())

        # in list labels we have the labels of this dataset
        list_labels = {item.item() for item in labels}

        to_be_sampled = []
        distribution_per_labels = {}
        # create the distribution for each class
        for label in list_labels:
            distribution = None
            if previous_distribution_per_labels:
                if label in previous_distribution_per_labels:
                    distribution = previous_distribution_per_labels[label]
            else:
                # For each label we want a distribution over the num_partitions
                distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
                distribution_per_labels[label] = distribution

            if distribution is not None:
                # we have to sample from the group of samples that have label equal
                # to label and not from the entire labels list.
                selected_labels = labels[labels == label]
                tmp_to_be_sampled = np.random.choice(
                    num_partitions, len(selected_labels), p=distribution[0]
                )
                # Inside to_be_sampled we save a counter for each label
                # The counter is the number of samples that we want to sample for each
                # partition
                to_be_sampled.append(Counter(tmp_to_be_sampled))
        # create the partitions
        partitions_index = {
            f"node_{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        public_private_indexes = {}

        for class_index, distribution_samples in zip(list_labels, to_be_sampled):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"node_{cluster_name}"] += index_per_label[
                    class_index
                ][:samples]

                index_per_label[class_index] = index_per_label[class_index][samples:]
        if max_size > 0:
            # split the data of each node in public and private
            for node_name, indexes in partitions_index.items():
                public_and_private = NonIIDPartition.do_iid_partitioning_with_indexes(
                    targets=labels[indexes],
                    data=dataset.data,
                    idx=torch.tensor(indexes),
                    num_partitions=2,
                    total_num_classes=total_num_classes,
                    alpha=1.0,
                )

                if max_size:
                    # We want that the public partition is at most a certain percentage of the
                    # entire dataset
                    total_sampled = len(labels[indexes])
                    max_size_public = int(max_size * total_sampled)
                    # If the maximum size of the public partition is less than the size of the
                    # public partition, we have to move some samples from the public partition
                    # to the private partition
                    if max_size_public < len(public_and_private[1]):
                        to_be_removed = len(public_and_private[1]) - max_size_public
                        # sample from the public partition to_be_removed samples, remove
                        # them from this partition and then, add them to the private partition
                        to_be_moved = np.random.choice(
                            public_and_private[1], to_be_removed, replace=False
                        )
                        public_and_private[1] = np.setdiff1d(
                            public_and_private[1], to_be_moved
                        )
                        public_and_private[0] = np.concatenate(
                            (public_and_private[0], to_be_moved)
                        )
                    # In the other case, we have to move some samples from the private partition
                    # to the public partition
                    else:
                        to_be_added = max_size_public - len(public_and_private[1])
                        # sample from the private partition to_be_added samples, remove
                        # them from this partition and then, add them to the public partition
                        to_be_moved = np.random.choice(
                            public_and_private[0], to_be_added, replace=False
                        )
                        public_and_private[0] = np.setdiff1d(
                            public_and_private[0], to_be_moved
                        )
                        public_and_private[1] = np.concatenate(
                            (public_and_private[1], to_be_moved)
                        )

                if previous_distribution_per_labels:
                    public_private_indexes[f"{node_name}"] = np.concatenate(
                        (public_and_private[0], public_and_private[1])
                    )
                else:
                    public_private_indexes[f"{node_name}_private"] = public_and_private[0]
                    public_private_indexes[f"{node_name}_public"] = public_and_private[1]
        elif max_size == 0:
            for node_name, indexes in partitions_index.items():
                public_private_indexes[f"{node_name}_private"] = indexes
        elif max_size == 1.0:
                public_private_indexes[f"{node_name}_private"] = indexes

        total = 0
        for cluster, samples in public_private_indexes.items():
            total += len(samples)

        if not previous_distribution_per_labels:
            assert total == len(labels)

        partitions_labels = {
            cluster: [item.item() for item in labels[samples]]
            for cluster, samples in public_private_indexes.items()
        }

        return public_private_indexes, partitions_labels, distribution_per_labels
    