from collections import Counter

import numpy as np
import torch


class NonIIDPartition:
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
        phase: str = None,
        alpha=1000000,
        previous_distribution_per_labels: dict = None,
    ) -> dict:
        if not alpha:
            raise ValueError("Alpha must be a positive number")
        labels = dataset.targets
        data = dataset.data if hasattr(dataset, "data") else dataset.samples
        name = "cluster_" if phase == "cluster" else "node_"
        num_labels = len(
            set(
                [
                    item.item() if isinstance(item, torch.Tensor) else item
                    for item in labels
                ]
            )
        )
        idx = torch.tensor(list(range(len(labels))))
        distribution_per_labels = {}

        index_per_label = {}
        for index, label in zip(idx, labels):
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in index_per_label:
                index_per_label[label] = []
            index_per_label[label].append(index.item())

        # in list labels we have the labels of this dataset
        list_labels = {
            item.item() if isinstance(item, torch.Tensor) else item for item in labels
        }

        to_be_sampled = []
        # create the distribution for each class
        for label in list_labels:
            labels = np.array(labels) if isinstance(labels, list) else labels
            # For each label we want a distribution over the num_partitions
            distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
            # we have to sample from the group of samples that have label equal
            # to label and not from the entire labels list.

            distribution = None
            if previous_distribution_per_labels:
                if label in previous_distribution_per_labels:
                    distribution = previous_distribution_per_labels[label]
            else:
                # For each label we want a distribution over the num_partitions
                distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
                distribution_per_labels[label] = distribution

            if distribution is not None:
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
            f"{name}{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        for class_index, distribution_samples in zip(list_labels, to_be_sampled):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"{name}{cluster_name}"] += index_per_label[
                    class_index
                ][:samples]

                index_per_label[class_index] = index_per_label[class_index][samples:]

        total = 0
        for cluster, samples in partitions_index.items():
            total += len(samples)

        assert total == len(labels)

        partitions_labels = {
            cluster: [
                item.item() if isinstance(item, torch.Tensor) else item
                for item in labels[samples]
            ]
            for cluster, samples in partitions_index.items()
        }

        if isinstance(data, list):
            data = np.array(data)
        partitions_data = {
            cluster: data[indexes] for cluster, indexes in partitions_index.items()
        }

        return (
            partitions_index,
            partitions_labels,
            partitions_data,
            distribution_per_labels,
        )

    def do_partitioning_double(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
        phase: str = None,
        alpha=1000000,
    ) -> dict:
        if not alpha:
            raise ValueError("Alpha must be a positive number")
        labels = dataset.targets
        data = dataset.data if hasattr(dataset, "data") else dataset.samples
        name = "cluster_" if phase == "cluster" else "node_"
        num_labels = len(
            set(
                [
                    item.item() if isinstance(item, torch.Tensor) else item
                    for item in labels
                ]
            )
        )
        idx = torch.tensor(list(range(len(labels))))

        index_per_label = {}
        for index, label in zip(idx, labels):
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in index_per_label:
                index_per_label[label] = []
            index_per_label[label].append(index.item())

        # in list labels we have the labels of this dataset
        list_labels = {
            item.item() if isinstance(item, torch.Tensor) else item for item in labels
        }

        to_be_sampled = []
        # create the distribution for each class
        for label in list_labels:
            labels = np.array(labels) if isinstance(labels, list) else labels
            # For each label we want a distribution over the num_partitions
            distribution = np.random.dirichlet(num_partitions // 2 * [alpha], size=1)
            # we have to sample from the group of samples that have label equal
            # to label and not from the entire labels list.
            selected_labels = labels[labels == label]
            tmp_to_be_sampled = np.random.choice(
                num_partitions // 2, len(selected_labels), p=distribution[0]
            )

            # Inside to_be_sampled we save a counter for each label
            # The counter is the number of samples that we want to sample for each
            # partition
            to_be_sampled.append(Counter(tmp_to_be_sampled))
        # create the partitions
        partitions_index = {
            f"{name}{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        mid = num_partitions // 2
        for class_index, distribution_samples in zip(list_labels, to_be_sampled):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"{name}{cluster_name}"] += index_per_label[
                    class_index
                ][: samples // 2]
                partitions_index[f"{name}{int(cluster_name)+mid}"] += index_per_label[
                    class_index
                ][samples // 2 : samples]

                index_per_label[class_index] = index_per_label[class_index][samples:]

        total = 0
        for cluster, samples in partitions_index.items():
            total += len(samples)

        assert total == len(labels)

        partitions_labels = {
            cluster: [
                item.item() if isinstance(item, torch.Tensor) else item
                for item in labels[samples]
            ]
            for cluster, samples in partitions_index.items()
        }

        if isinstance(data, list):
            data = np.array(data)
        partitions_data = {
            cluster: data[indexes] for cluster, indexes in partitions_index.items()
        }

        return partitions_index, partitions_labels, partitions_data

    def do_iid_partitioning_with_indexes(
        targets: torch.Tensor,
        data: torch.Tensor,
        idx: torch.Tensor,
        num_partitions: int,
        total_num_classes: int,
        phase: str = None,
        alpha=1000000,
    ) -> dict:
        if not alpha:
            raise ValueError("Alpha must be a positive number")

        name = "cluster_" if phase == "cluster" else "node_"
        num_labels = len(
            set(
                [
                    item.item() if isinstance(item, torch.Tensor) else item
                    for item in targets
                ]
            )
        )
        # idx = torch.tensor(list(range(len(labels))))

        index_per_label = {}
        for index, label in zip(idx, targets):
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in index_per_label:
                index_per_label[label] = []
            index_per_label[label].append(index.item())

        # in list labels we have the labels of this dataset
        list_labels = {
            item.item() if isinstance(item, torch.Tensor) else item for item in targets
        }

        to_be_sampled = []
        # create the distribution for each class
        for label in list_labels:
            # labels = np.array(labels) if isinstance(labels, list) else labels
            # For each label we want a distribution over the num_partitions
            distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
            # we have to sample from the group of samples that have label equal
            # to label and not from the entire labels list.
            selected_labels = targets[targets == label]
            tmp_to_be_sampled = np.random.choice(
                num_partitions, len(selected_labels), p=distribution[0]
            )
            # Inside to_be_sampled we save a counter for each label
            # The counter is the number of samples that we want to sample for each
            # partition
            to_be_sampled.append(Counter(tmp_to_be_sampled))
        # create the partitions
        partitions_index = {
            f"{name}{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        for class_index, distribution_samples in zip(list_labels, to_be_sampled):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"{name}{cluster_name}"] += index_per_label[
                    class_index
                ][:samples]

                index_per_label[class_index] = index_per_label[class_index][samples:]

        total = 0
        for cluster, samples in partitions_index.items():
            total += len(samples)

        assert total == len(targets)

        partitions = []
        for node_name in partitions_index:
            partitions.append(partitions_index[node_name])
        return partitions
