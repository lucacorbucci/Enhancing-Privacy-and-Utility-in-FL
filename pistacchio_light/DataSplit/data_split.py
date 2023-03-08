import copy
import random
from collections import Counter, defaultdict
from random import shuffle
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from pistacchio_simulator.DataSplit.custom_dataset import MyDataset
from pistacchio_simulator.Exceptions.errors import InvalidSplitConfigurationError
from pistacchio_simulator.Utils.task import Task, TaskType


random.seed(42)
np.random.seed(0)


class DataSplit:
    """This class is used to split the dataset in different ways."""

    @staticmethod
    def convert_subset_to_dataset(subset: Subset) -> tuple[list, list]:
        """This function converts a subset of data to a dataset.

        Args:
            subset (_type_): the subset of data to be converted to a dataset
        Returns:
            Tuple[List, List]: the converted dataset
        """
        datapoint = []
        targets = []
        for sample in subset:
            datapoint.append(sample)
            targets.append(sample[-1])
        return datapoint, targets

    @staticmethod
    def create_splits(
        dataset: torch.utils.data.Dataset,
        num_workers: int,
        max_samples_per_cluster: list | int | None = None,
    ) -> tuple[list[int], list[int]]:
        """This function returns a list that will be used to split the dataset.
        If the parameter max_samples_per_cluster is equal to -1 then it considers
        the size of the dataset and then splits it in equal parts.
        Otherwise, if we pass a value different than -1, it will create
        a splitting configuration where each cluster will receive
        max_samples_per_cluster samples.

        Raises
        ------
            ValueError: it raises this error when we choose
            a value for max_samples_per_cluster that is too big.
            In particular, if max_samples_per_cluster * num_workers > len(dataset)
            then the function will raise an exception.
            It can also raise ValueError when the dataset passed as
            parameter is empty.

        Args:
            dataset (torch.utils.data.Dataset): the dataset we want to split
            num_workers (int): number of nodes
            max_samples_per_cluster (int): default value == -1. The maximum
            amount of samples per cluster

        Example:
            >>> dataset = MyDataset(
                    samples=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100,
                    targets=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 100,
                )
            >>>

        Returns
        -------
            Tuple[List[int], List[int]]: the first list is a list of sizes of the splits
            the second list can be empty if max_samples_per_cluster == -1 or it contains
            the remaining data that could not be distributed equally among the clusters.
        """
        num_targets = len(dataset.targets)
        remaining_data: list[int] = []

        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        if not max_samples_per_cluster:
            split_size = int(num_targets / num_workers)
            rem = num_targets - (split_size * num_workers)
            split = [split_size for _ in range(num_workers)]
            while rem > 0:
                split[random.randint(0, num_workers - 1)] += 1
                rem -= 1
        else:
            if isinstance(max_samples_per_cluster, list):
                if sum(max_samples_per_cluster) > len(dataset.targets):
                    raise ValueError("Max samples per cluster is too high.")
                if len(max_samples_per_cluster) != num_workers:
                    raise ValueError(
                        "The length of max_samples_per_cluster must be equal to \
                            num_workers",
                    )
                split = max_samples_per_cluster
                rem = num_targets - sum(max_samples_per_cluster)
            elif isinstance(max_samples_per_cluster, int):
                if max_samples_per_cluster * num_workers > len(dataset.targets):
                    raise ValueError("Max samples per cluster is too high.")
                rem = num_targets - (max_samples_per_cluster * num_workers)
                split = [max_samples_per_cluster for _ in range(num_workers)]
            remaining_data.append(0)
            while rem > 0:
                remaining_data[0] += 1
                rem -= 1

        return split, remaining_data

    @staticmethod
    def split_dataset(
        dataset: torch.utils.data.Dataset,
        num_workers: int,
        max_samples_per_cluster: list | int | None = None,
    ) -> Any:
        """This function splits the dataset in equal parts. If max_samples_per_cluster
        is different than -1 then it will split the dataset in a way that each cluster
        will receive max_samples_per_cluster samples.

        Args:
            dataset (torch.utils.data.Dataset): the dataset we want to split
            num_workers (int): number of nodes of the dataset

        Returns
        -------
            List[torch.utils.data.DataLoader]: the splitted dataset
        """
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            datapoint, targets = DataSplit.convert_subset_to_dataset(dataset)
            dataset = MyDataset(datapoint, targets)
        split, remaining_data = DataSplit.create_splits(
            dataset,
            num_workers,
            max_samples_per_cluster,
        )

        # if we have remaining data we will create a final artificial cluster.
        # It is necessary to do this because the function random_split
        # does not accept a list of sizes that does not sum up to the
        # length of the dataset
        if len(remaining_data) > 0:
            split.append(remaining_data[0])

        return torch.utils.data.random_split(
            dataset,
            split,
            generator=torch.Generator().manual_seed(42),
        )

    @staticmethod
    def merge_indices(datasets: list) -> torch.utils.data.DataLoader:
        """This function merges the indices of a list of datasets
        into a single list.

        Args:
            datasets (list): the list of datasets to merge

        Returns
        -------
            list: the merged list of indices
        """
        indices = []
        for dataset in datasets:
            indices += dataset.indices
        return indices

    @staticmethod
    def check_percentage_validity(
        data: torch.utils.data.DataLoader,
        percentage_configuration: dict,
    ) -> None:
        """This function checks if the percentage configuration is valid.

        Args:
            data (torch.utils.data.DataLoader): data we want to split
            percentage_configuration (dict): configuration of the percentage

        Raises
        ------
            InvalidSplitConfigurationError: It is raised when the percentage is not valid
        """
        num_classes = len(data.classes)
        percentages: Counter = Counter()
        for cluster_distribution in percentage_configuration.values():
            for key, value in cluster_distribution.items():
                percentages[key] += value
        if sum(percentages.values()) != num_classes * 100 and not all(
            i == 100 for i in percentages.values()
        ):
            raise InvalidSplitConfigurationError()

    @staticmethod
    def generate_classes_dictionary(data: torch.utils.data.DataLoader) -> dict:
        """Generates a dictionary with all the possible classes as key and for
        each key all the possible values
        {"class_0": [data]}.

        Args:
            data (torch.utils.data.DataLoader): data we want to split

        Returns
        -------
            dict: _description_
        """
        if isinstance(data, torch.utils.data.DataLoader):
            data = data.dataset
        classes_dictionary: dict = {}
        index = 0
        classes_dictionary = defaultdict(list, classes_dictionary)

        for _, current_target in zip(data, data.targets):
            classes_dictionary[
                current_target.item()
                if isinstance(current_target, torch.Tensor)
                else current_target
            ].append(index)
            index += 1

        return classes_dictionary

    @staticmethod
    def generate_aggregated_percentages(data: list) -> dict:
        """Generates a dictionary with all the possible classes and
        for each class the corresponding aggregated percentages
        {"class_0": {"user_0": 40, "user_1": 20, "user_2":40}}.

        Args:
            data (list): percentage distribution

        Returns
        -------
            dict: the new aggregated dictionary
        """
        percentage_dictionary = {}
        for cluster_name in data:
            for class_name, percentage in cluster_name[1].items():
                if class_name not in percentage_dictionary:
                    percentage_dictionary[class_name] = {cluster_name[0]: percentage}
                else:
                    percentage_dictionary[class_name][cluster_name[0]] = percentage
        return percentage_dictionary

    @staticmethod
    def sample_list_by_percentage(class_percentage: dict, indices: list) -> dict:
        """This function split the list indices passed as parameter
        in N parts. The size of each part depends on the percentage
        that we specify in class_percentage and that correspond to
        each user.

        Args:
            class_percentage (_type_): percentage of data of each user
            indices (_type_): list of indices we want to split among users
        """
        np.random.shuffle(indices)
        percentages = np.array([value / 100 for value in class_percentage.values()])
        split_idx = np.r_[0, (len(indices) * percentages.cumsum()).astype(int)]
        out = [indices[i:j] for (i, j) in zip(split_idx[:-1], split_idx[1:])]

        output = dict(zip(class_percentage.keys(), out))
        return output

    @staticmethod
    def generate_aggregated_indices(percentages: dict, indices: dict) -> dict:
        """# Inside samples we have the classes as keys and the
        # corresponding dict of users and indices as values
        # Example: {0: {"user_1": [...], "user_2": [...], "user_3": [...]}}.

        Args:
            percentages (dict): _description_
            indices (dict): _description_

        Returns
        -------
            dict: _description_
        """
        samples = {}
        for class_name, class_percentage in percentages.items():
            samples[class_name] = DataSplit.sample_list_by_percentage(
                class_percentage,
                indices[class_name],
            )

        return samples

    @staticmethod
    def aggregate_indices_by_cluster(samples: dict) -> dict:
        """This function aggregates the indices by cluster.
        Originally, they are aggregated by class, we want to merge the
        N lists of indices for each class in a single list of indices.

        Example: {"cluster_0": [1,2,3,4,5,6.....]}

        Args:
            samples (dict): a dictionary with all the indices of each cluster
            divided by class

        Returns
        -------
            dict: a dictionary with all the indices of each cluster
        """
        splitted_indices = {}
        for _, values in samples.items():
            for cluster_name, cluster_indices in values.items():
                if cluster_name not in splitted_indices:
                    splitted_indices[cluster_name] = cluster_indices
                else:
                    splitted_indices[cluster_name] += cluster_indices
                random.shuffle(splitted_indices[cluster_name])
        return splitted_indices

    @staticmethod
    def split_cluster_dataset_in_parts(
        data: torch.utils.data.DataLoader,
        cluster_names: list,
        splitted_indices: dict,
    ) -> tuple[list, list]:
        """This function takes as input the dataset that we want to split,
        a list with the names of the clusters and a dictionary where we store
        as key the cluster names and as values a list of N lists. Each
        of these lists contains the indices of the dataset that we
        want to assign to the each user.

        Args:
            data (torch.utils.data.DataLoader): the dataset we want to split
            cluster_names (list): the names of the clusters
            splitted_indices (dict): the dictionary with the indices of each cluster

        Example:
        >>> cluster_names = ["cluster1", "cluster2"]
        >>> splitted_indices = {
            "cluster1": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]],
            "cluster2": [[11, 12], [13, 14]],
        }
        >>> mock_dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
        >>> cluster_datasets, counter = DataSplit.split_cluster_dataset_in_parts(
            mock_dataloader, cluster_names, splitted_indices
        )

        Returns
        -------
            Tuple[list, list]: the first element of the tuple is a list of
            torch.utils.data.Subset objects, each of them is the subset
            that is assigned to a user. The second element is a list of
            Counters, each of them contains the number of elements of each
            class that are assigned to a user.
            first element: [<torch.utils.data.dataset.Subset object at 0x155314c70>,
                <torch.utils.data.dataset.Subset object at 0x155314cd0>,
                <torch.utils.data.dataset.Subset object at 0x155314d30>,
                <torch.utils.data.dataset.Subset object at 0x155314d90>,
                <torch.utils.data.dataset.Subset object at 0x155314df0>]
            second element: [Counter({0: 4}), Counter({1: 4}),
                Counter({2: 3}), Counter({0: 2}), Counter({1: 2})]

        """
        targets = np.array(data.targets)
        counter = []
        cluster_datasets = []
        for name in cluster_names:
            indices = splitted_indices[name]
            for user_indices in indices:
                cluster_datasets.append(torch.utils.data.Subset(data, user_indices))
                counter.append(Counter(targets[user_indices]))
        return cluster_datasets, counter

    @staticmethod
    def aggregate_indices_by_class(targets_cluster: list, indices: list) -> dict:
        """Aggregates the indices assigned to the cluster by class.
        Returns a dictionary with the indices of each class.

        Args:
            targets_cluster (list): _description_
            indices (list): _description_

        Example:
        >>> splitted_indices = [1, 2, 3, 4, 5, 6]
        >>> targets = [0, 1, 2, 0, 0, 1]
        >>> aggregated_indices = DataSplit.aggregate_indices_by_class(targets,
            splitted_indices)
        {0: [1, 4, 5], 1: [2, 6], 2: [3]}

        Returns
        -------
            dict: A dictionary with the iaggregated_percentagesndices of each class
            {"class_0": [1,2,3,4,5,6,7,8,9,10],
            "class_1": [11,12,13,14,15,16,17,18,19,20]}
        """
        indices_per_class = {}
        for target, index in zip(targets_cluster, indices):
            if target not in indices_per_class:
                indices_per_class[target] = [index]
            else:
                indices_per_class[target].append(index)
        return indices_per_class

    @staticmethod
    def create_percentage_subsets(
        splitted_indices: dict,
        nodes_names: list,
        targets: np.ndarray,
        data: torch.utils.data.DataLoader,
    ) -> tuple[list, list]:
        """This function creates the subsets of the original dataset
        for a cluster. Given the dataset, it is splitted in N parts
        where N is the number of users in the cluster.

        Example:
        >>> splitted_indices = {
            "cluster_0_user_0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "cluster_0_user_1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "cluster_0_user_2": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "cluster_0_user_3": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            "cluster_0_user_4": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            }
        >>> nodes_names = [
            "cluster_0_user_0",
            "cluster_0_user_1",
            "cluster_0_user_2",
            "cluster_0_user_3",
            "cluster_0_user_4",
            ]
        >>> data = DataLoader(dataset, batch_size=1, shuffle=True)
        >>> cluster_datasets, counter = DataSplit.create_percentage_subsets(
            data=data,
            splitted_indices=splitted_indices,
            nodes_names=nodes_names,
            targets=np.array(targets),
            )
        ([<torch.utils.data.dataset.Subset object at 0x140689640>,
         <torch.utils.data.dataset.Subset object at 0x140689550>, ...],
        [Counter({4: 2, 0: 2, 8: 2, 5: 1, 3: 1, 6: 1, 1: 1}),
        Counter({1: 3, 4: 2, 7: 2, 0: 2, 3: 1}), ..... )])

        Args:
            splitted_indices (dict): is a dictionary
            with the names of the nodes as keys and the lists of the indices
            that will be assigned to each node as values.
            nodes_names (list): contains the names of the users in the cluster.
            targets (np.ndarray): contains the targets of the dataset.
            data (torch.utils.data.DataLoader): the original dataset we want to split

        Returns
        -------
            Tuple[list, list]: a list that contains the subsets created
            from the original dataset and assigned to each user in the cluster and
            a list that contains the counters of the targets of each subset.
        """
        counter = []
        cluster_datasets = []
        for name in nodes_names:
            indices = splitted_indices[name]
            cluster_datasets.append(torch.utils.data.Subset(data, indices))
            counter.append(Counter(targets[indices]))
        return cluster_datasets, counter

    @staticmethod
    def split_cluster_dataset_by_percentage(
        nodes_distribution: dict,
        data: torch.utils.data.DataLoader,
        splitted_indices: dict,
    ) -> tuple[list, list]:
        """This function is used to split the dataset in N parts
        where N is the number of clusters. Then each part
        is splitted in M parts where M is the number of users.

        Example:
        >>> nodes_distribution = {
            "cluster_0": {
                "user_0": {0: 40, 1: 20},
                "user_1": {0: 40, 1: 40},
                "user_2": {0: 20, 1: 40},
            },
            "cluster_1": {
                "user_0_1": {0: 40, 1: 20},
                "user_1_1": {0: 40, 1: 40},
                "user_2_1": {0: 20, 1: 40},
            },
            }
        >>> splitted_indices = {
            "cluster_0": [0, 1, 2, 3, 4],
            "cluster_1": [5, 6, 7, 8, 9],
            }
        >>> data_loader = DataLoader(mock_dataset, batch_size=1)
        >>> cluster_datasets, counters = DataSplit.split_cluster_dataset_by_percentage(
                nodes_distribution, data_loader, splitted_indices
            )
        ([<torch.utils.data.dataset.Subset object at 0x14491a100>,
            <torch.utils.data.dataset.Subset object at 0x14491a130>, .....],
            [Counter(), Counter({0: 1, 1: 1}), Counter({1: 2, 0: 1}),....)])

        Args:
            nodes_distribution (dict): the configuration that we want to
            use in each cluster and for each user.
            data (torch.utils.data.DataLoader): the dataset we want to split.
            splitted_indices (dict): a dictionary with the names of the clusters
            as keys and the lists of the indices that will be assigned to each
            cluster as values.

        Returns
        -------
            Tuple[list, list]: a list that contains the subsets created
            from the original dataset and assigned to each user in the cluster and
            a list that contains the counters of the targets of each subset.
        """
        targets = np.array(data.targets)
        cluster_datasets, counters = [], []
        for cluster_name, indices in splitted_indices.items():
            # Indices of the cluster aggregated by class
            # Example {"class_0": [1,2,3,4,5,6,7,8,9,10],
            #          "class_1": [11,12,13,14,15,16,17,18,19,20]}
            indices_per_class = DataSplit.aggregate_indices_by_class(
                targets_cluster=targets[indices],
                indices=indices,
            )

            percentages = DataSplit.generate_aggregated_percentages(
                list(nodes_distribution[cluster_name].items()),
            )
            nodes_names = nodes_distribution[cluster_name].keys()

            samples = DataSplit.generate_aggregated_indices(
                percentages=percentages,
                indices=indices_per_class,
            )

            # Example: {"cluster_0": [1,2,3,4,5,6.....]}
            splitted_indices = DataSplit.aggregate_indices_by_cluster(samples=samples)
            cluster_datasets_tmp, counters_tmp = DataSplit.create_percentage_subsets(
                targets=targets,
                nodes_names=nodes_names,
                splitted_indices=splitted_indices,
                data=data,
            )
            cluster_datasets += cluster_datasets_tmp
            counters += counters_tmp
        return cluster_datasets, counters

    @staticmethod
    def random_class_distribution(num_nodes: int, total_sum: int) -> list:
        """This function generates a random distribution of percentages
        for the classes of the cluster. It considers the number of the nodes in
        the cluster and splits the data in N parts where N is the number of nodes.
        Example: in cluster 0 we have user 0 with the following distribution:
        [40, 0, 20, 20, 0, 0, 0, 20]
        When the percentage value is 0, we will not consider that class for that user.


        Args:
            num_nodes (int): number of nodes of the cluster
            total_sum (int): total sum of the percentages of the classes

        Returns
        -------
            _type_: _description_
        """
        class_distribution = [0] * num_nodes
        values = [2, 5, 8]
        while sum(class_distribution) < total_sum:
            random_value = np.random.choice(values)
            random_index = np.random.randint(0, num_nodes)
            if sum(class_distribution) + random_value <= total_sum:
                class_distribution[random_index] += random_value
            else:
                class_distribution[random_index] += total_sum - sum(class_distribution)

        assert total_sum == sum(class_distribution)
        return class_distribution

    @staticmethod
    def generate_nodes_distribution(num_nodes: int, classes: list, names: list) -> dict:
        """This function generates the final distribution of the classes of the cluster.
        In particular, we consider the number of nodes in the cluster and the number of
        classes assigned to the cluster.
        For each of the classes, we assign a certain percentage of data for each node.
        Every time we generate a distribution, we check if we have a node without any
        class assignment. If this happens, we generate a new distribution.

        Args:
            num_nodes (int): number of nodes in the cluster
            classes (list): classes assigned to the cluster
            names (list): names of the nodes inside the cluster

        Returns
        -------
            _type_: _description_
        """
        nodes_distribution: dict = {}
        for name in names:
            nodes_distribution[name] = {}
        stop = False

        while not stop:
            for class_ in classes:
                class_distribution = DataSplit.random_class_distribution(
                    num_nodes=num_nodes,
                    total_sum=100,
                )
                for node_name, item in zip(names, class_distribution):
                    if item != 0:
                        nodes_distribution[node_name][class_] = item
            stop = True
            for node_name in nodes_distribution.values():
                if len(node_name) == 0:
                    stop = False
        return nodes_distribution

    @staticmethod
    def remove_indices_from_dataset(
        dataset: torch.utils.data.Dataset,
        indices_to_remove: np.ndarray,
    ) -> None:
        """This function removes the indices passed as
        parameter from a dataset.

        Args:
            dataset (torch.utils.data.Dataset): the dataset we want to modify
            indices_to_remove (list): the indices we want to remove from the dataset
        """
        samples = np.array(dataset.data)
        targets = np.array(dataset.targets)
        mask = np.ones(len(dataset.targets), dtype=bool)
        mask[indices_to_remove] = False
        dataset.targets = torch.tensor(targets[mask])
        dataset.data = torch.tensor(samples[mask])

    ## -------------- Functions to reduce the samples of the dataset -------------- ##

    @staticmethod
    def reduce_samples(
        dataset: torch.utils.data.Dataset,
        classes: list,
        percentage_underrepresented_classes: list | None = None,
        num_samples_underrepresented_classes: list | None = None,
    ) -> None:
        """This function reduces the samples of the classes indicated
        in the parameters.

        Args:
            dataset (torch.utils.data.Dataset): the dataset we want to reduce
            classes (list): the classes we want to reduce
            percentage_underrepresented_classes (list, optional): the percentage of
            samples of the specified class that we want to remove
            from the dataset . Defaults to []. num_samples_underrepresented_classes
             (list, optional): The amount of samples that
            we want for each of the classes we want to reduce. Defaults to [].
        """
        if (
            percentage_underrepresented_classes is None
            and num_samples_underrepresented_classes is None
        ):
            raise ValueError(
                "You have to specify either the percentage or the number of samples",
            )

        if percentage_underrepresented_classes is not None:
            DataSplit.reduce_samples_by_percentage(
                dataset=dataset,
                classes=classes,
                percentage_underrepresented_classes=percentage_underrepresented_classes,
            )
        else:
            assert num_samples_underrepresented_classes
            DataSplit.reduce_samples_by_num_samples(
                dataset=dataset,
                classes=classes,
                num_samples_underrepresented_classes=num_samples_underrepresented_classes,
            )

    @staticmethod
    def reduce_samples_by_percentage(
        dataset: torch.utils.data.Dataset,
        classes: list,
        percentage_underrepresented_classes: list,
    ) -> None:
        """This function reduces the samples of the classes indicated
        in the list classes by the percentages indicated in the list
        percentage_underrepresented_classes.

        Args:
            dataset (torch.utils.data.Dataset): the dataset we want to reduce
            classes (list): the classes we want to reduce
            percentage_underrepresented_classes (list): the percentage of samples
            of the specified class that we want to remove from the dataset
        """
        # I want to add an attribute indices to the dataset so that I can
        # use this in the stratified sampling
        dataset.indices = np.arange(len(dataset.targets))

        for class_, percentage in zip(classes, percentage_underrepresented_classes):
            targets = np.array(dataset.targets)
            indices = np.arange(len(dataset.targets))

            indices_class = indices[targets == class_]
            num_samples = int(len(indices_class) * percentage)
            indices_to_remove = np.random.choice(
                indices_class,
                num_samples,
                replace=False,
            )

            indices = np.setdiff1d(indices, indices_to_remove)

            DataSplit.remove_indices_from_dataset(dataset, indices_to_remove)

    @staticmethod
    def reduce_samples_by_num_samples(
        dataset: torch.utils.data.Dataset,
        classes: list,
        num_samples_underrepresented_classes: list,
    ) -> None:
        """This function reduces the samples of the classes indicated
        in the list classes to the number of samples indicated in the list
        num_samples_underrepresented_classes.

        Args:
            dataset (torch.utils.data.Dataset): the dataset we want to reduce
            classes (list): the classes we want to reduce
            num_samples_underrepresented_classes (list): the amount of samples
            of the specified class that we want to keep in the dataset
        """
        # I want to add an attribute indices to the dataset so that I can
        # use this in the stratified sampling
        dataset.indices = np.arange(len(dataset.targets))

        for class_, num_samples_underrepresented_class in zip(
            classes,
            num_samples_underrepresented_classes,
        ):
            targets = np.array(dataset.targets)
            indices = np.arange(len(dataset.targets))

            indices_class = indices[targets == class_]
            if num_samples_underrepresented_class > len(indices_class):
                raise ValueError(
                    "The number of samples to keep is greater \
                        than the number of samples of the class",
                )
            num_samples = len(indices_class) - num_samples_underrepresented_class
            indices_to_remove = np.random.choice(
                indices_class,
                num_samples,
                replace=False,
            )

            indices = np.setdiff1d(indices, indices_to_remove)

            DataSplit.remove_indices_from_dataset(dataset, indices_to_remove)

    ## -------------- Functions for Stratified Sampling -------------- ##

    @staticmethod
    def stratified_sampling(
        dataset: torch.utils.data.DataLoader,
        num_workers: int,
        max_samples_per_cluster: list | int | None = None,
    ) -> tuple[Any, list[Counter]]:
        """This function performs stratified sampling on the dataset.

        Args:
            dataset (torch.utils.data.DataLoader): the dataset to sample from
            num_workers (int): number of users that will be created


        Returns
        -------
            Tuple[Any, Counter]: The splitted dataset and a counter with the
            number of samples per cluster
        """
        splitted_datasets = DataSplit.split_dataset(
            dataset=dataset,
            num_workers=num_workers,
            max_samples_per_cluster=max_samples_per_cluster,
        )

        dataset_counter = []

        for splitted_dataset in splitted_datasets:
            if isinstance(splitted_dataset, torch.utils.data.dataset.Subset):
                _, targets = DataSplit.convert_subset_to_dataset(splitted_dataset)
                dataset_counter.append(Counter(targets))

        return splitted_datasets, dataset_counter

    @staticmethod
    def stratified_sampling_with_some_nodes_reduced(
        dataset: torch.utils.data.DataLoader,
        num_workers: int,
        num_reduced_nodes: int,
        max_samples_per_cluster: int,
        underrepresented_classes: list[int],
        percentage_underrepresented_classes: list[float],
    ) -> tuple[Any, list[Counter]]:
        """This function performs stratified sampling on the dataset.

        Args:
            dataset (torch.utils.data.DataLoader): the dataset to sample from
            num_workers (int): number of users that will be created
            num_reduced_nodes (int): number of users that will have a reduced dataset
            max_samples_per_cluster (int): maximum number of samples per node.

        Returns
        -------
            Tuple[Any, Counter]: The splitted dataset and a counter with the
            number of samples per cluster
        """
        if num_reduced_nodes > num_workers:
            raise ValueError(
                "Number of reduced nodes cannot be greater than number of workers",
            )

        splitted_datasets = DataSplit.split_dataset(
            dataset=dataset,
            num_workers=num_workers - num_reduced_nodes,
            max_samples_per_cluster=max_samples_per_cluster,
        )

        splitted_datasets = splitted_datasets[:-1]
        index_list = DataSplit.merge_indices(splitted_datasets)
        new_dataset = copy.deepcopy(dataset)
        DataSplit.remove_indices_from_dataset(new_dataset, index_list)

        DataSplit.reduce_samples_by_percentage(
            new_dataset,
            underrepresented_classes,
            percentage_underrepresented_classes,
        )

        splitted_datasets_reduced = DataSplit.split_dataset(
            dataset=new_dataset,
            num_workers=num_reduced_nodes,
            max_samples_per_cluster=max_samples_per_cluster,
        )

        splitted_datasets += splitted_datasets_reduced

        dataset_counter = []

        for splitted_dataset in splitted_datasets:
            if isinstance(splitted_dataset, torch.utils.data.dataset.Subset):
                _, targets = DataSplit.convert_subset_to_dataset(splitted_dataset)
                dataset_counter.append(Counter(targets))

        return splitted_datasets, dataset_counter

    ## -------------- Functions for Percentage Sampling -------------- ##

    @staticmethod
    def percentage_split(
        dataset: torch.utils.data.DataLoader,
        percentage_configuration: dict,
        task: Task,
        num_workers: int = 0,
        nodes_distribution: dict | None = None,
    ) -> tuple[list, list]:
        """This function is used to split the original dataset based on a
        configuration passed as a parameter.
        We can split the dataset in two ways:
        - If we don't provide the paramert nodes_distribution, we
        will split the dataset in N parts (N is the number of cluster)
        and then for each cluster we will split again the dataset in
        M parts (M is the number of users in the cluster). In this case
        the distribution will be non iid for the clusters
        and iid for the users.
        - If we provide a nodes_distribution, we will split the dataset
        in N parts (N is the number of cluster) and then for each cluster
        we will split again the dataset in M parts (M is the number of users
        in the cluster). In this case we will use a percentage split
        both for the clusters and for the users. If we provide a proper
        nodes_distribution dictionary, we will have both a non iid
        distribution for the clusters and a non iid distribution for the users.

        Example:
        >>> percentage_configuration = {
            "cluster_0": {0: 60, 1: 30, 2: 20, 3: 20},
            "cluster_1": {1: 70, 2: 40, 3: 20, 4: 20},
            "cluster_2": {2: 40, 3: 20, 4: 20, 5: 20},
            "cluster_3": {3: 40, 4: 20, 5: 20, 6: 30},
            "cluster_4": {4: 40, 5: 20, 6: 30, 7: 10},
            "cluster_5": {5: 40, 6: 20, 7: 30, 8: 30},
            "cluster_6": {6: 20, 7: 40, 8: 30, 9: 70},
            "cluster_7": {7: 20, 8: 40, 9: 30, 0: 40},
        }
        >>> nodes_distribution = {
            "cluster_0": {
                "cluster_0_user_0": {0: 18, 1: 15, 2: 19, 3: 36},
                "cluster_0_user_1": {0: 9, 1: 40, 2: 5, 3: 8},
                "cluster_0_user_2": {0: 30, 1: 18, 2: 20, 3: 38},
                "cluster_0_user_3": {0: 28, 1: 17, 2: 27, 3: 10},
                "cluster_0_user_4": {0: 15, 1: 10, 2: 29, 3: 8},
            },
            "cluster_1": {
                "cluster_1_user_0": {1: 23, 2: 8, 3: 28, 4: 23},
                "cluster_1_user_1": {1: 30, 2: 10, 3: 12, 4: 23},
                "cluster_1_user_2": {1: 25, 2: 21, 3: 29, 4: 15},
                "cluster_1_user_3": {1: 2, 2: 25, 3: 10, 4: 28},
                "cluster_1_user_4": {1: 20, 2: 36, 3: 21, 4: 11},
            }, .....
        }
        >>> samples = list(np.random.rand(10000, 10))
        >>> targets = list(np.random.randint(0, 10, size=10000))
        >>> mock_dataset = MyDataset(samples, targets)
        >>> cluster_datasets, counter = DataSplit.percentage_split(
            data=data,
            percentage_configuration=percentage_configuration,
            num_workers=2,
            nodes_distribution=nodes_distribution,
            task="federated"
        )
        ([<torch.utils.data.dataset.Subset object at 0x14b0434c0>,
            <torch.utils.data.dataset.Subset object at 0x14b043520>, .....],
        [Counter({0: 35, 3: 24, 1: 15, 2: 14}),
        Counter({1: 40, 0: 18, 3: 5, 2: 4}), ......])

        Args:
            data (torch.utils.data.DataLoader): the dataset we want to split
            percentage_configuration (dict): the configuration of the cluster we
            want to create
            num_workers (int, optional): _description_. Defaults to 0. The number of
            users per cluster
            nodes_distribution (dict, optional): _description_. Defaults to None. The
            distribution of the users per cluster
            task (Task): the task we want to perform after the split. It is important
            to specify it because for some tasks we don't want to check the percentage
            validity of the configuration passed as parameter.

        Returns
        -------
            Tuple[list, list]: the first returned element is the list with
            the subsets that were created during the process. The second
            returned element is a list with the counters for each cluster,
            this is useful to understand how many samples are assigned to each user.
        """
        # Check if the percentage split given as input is valid or not
        if task.task_type != TaskType.FAIRNESS:
            DataSplit.check_percentage_validity(dataset, percentage_configuration)

        # get the names of the clusters
        cluster_names = list(percentage_configuration.keys())

        # {"class_0": [data]} A dictionary with all the classes and
        # the corresponding data
        classes_dictionary = DataSplit.generate_classes_dictionary(dataset)
        # {"class_0": [0,1,2,....n]} A key for each class and the
        # corresponding list of indices
        indices = classes_dictionary

        percentages = DataSplit.generate_aggregated_percentages(
            list(percentage_configuration.items()),
        )

        data: dict = {}
        data = defaultdict(list, data)
        samples = DataSplit.generate_aggregated_indices(
            percentages=percentages,
            indices=indices,
        )

        # Example: {"cluster_0": [1,2,3,4,5,6.....]}
        splitted_indices = DataSplit.aggregate_indices_by_cluster(samples=samples)
        if nodes_distribution:
            cluster_datasets, counter = DataSplit.split_cluster_dataset_by_percentage(
                nodes_distribution=nodes_distribution,
                data=dataset,
                splitted_indices=splitted_indices,
            )
        else:
            # Now we want to split the indices of each cluster in N parts
            # where N is the number of users in the cluster
            # We have {"user_0": [[0,1,2], [4,5,6]]}
            for name in cluster_names:
                splitted_indices[name] = np.array_split(
                    np.array(splitted_indices[name]),
                    num_workers,
                )

            cluster_datasets, counter = DataSplit.split_cluster_dataset_in_parts(
                data=dataset,
                splitted_indices=splitted_indices,
                cluster_names=cluster_names,
            )

        return cluster_datasets, counter

    @staticmethod
    def percentage_sampling_max_samples(
        dataset: torch.utils.data.DataLoader,
        percentage_configuration: dict,
        num_workers: int,
        max_samples_per_cluster: list | int,
        num_nodes: int = 1,
    ) -> tuple[list, list]:
        """This method is used to split the dataset using the percentage sampling.
        For each cluster we specify the percentage of samples we want in the cluster.
        For instance if we have 10 classes we could specify that we want 10% of each
        class in cluster 0. The percentage is specified as a dictionary where the key
        is the name of the cluster and the value is a dictionary with the percentage
        of each class. For instance:
            "percentage_configuration": {
                "cluster_0": {
                    "0": 10,
                    "1": 10,
                    "2": 10,
                    "3": 10,
                    "4": 10,
                    "5": 10,
                    "6": 10,
                    "7": 10,
                    "8": 10,
                    "9": 10
                },
            }.

        Example:


        Args:
            dataset (torch.utils.data.DataLoader): the dataset we want to split
            percentage_configuration (dict): the configuration of the cluster
             we want to create
            num_workers (int): total number of clusters we want to have
            max_samples_per_cluster (Union[list, int]): maximum number of samples
            we want inside each cluster this can be a list when we want to specify
             the size of each cluster or a int when we specify
            a size that is the same for all the clusters

        Raises
        ------
            ValueError: When the length of max_samples_per_cluster is
            not equal to num_workers
            ValueError: When the sum of max_samples_per_cluster is greater
            than the length of the dataset
        """
        if isinstance(max_samples_per_cluster, int):
            max_samples_per_cluster = [max_samples_per_cluster] * num_workers

        if len(max_samples_per_cluster) != num_workers:
            raise ValueError(
                "The length of max_samples_per_cluster must be equal to num_workers",
            )

        if sum(max_samples_per_cluster) > len(dataset):
            raise ValueError(
                "The sum of max_samples_per_cluster must be less \
                    than the length of the dataset",
            )
        for cluster_name in percentage_configuration:
            tmp_sum = 0
            for _, value in percentage_configuration[cluster_name].items():
                if value > 100:
                    raise ValueError(
                        f"The percentage of each class must be less \
                             than 100, but {value} was given",
                    )
                tmp_sum += value
            if tmp_sum != 100:
                raise ValueError(
                    f"The sum of the percentages of each class \
                         must be equal to 100, but {tmp_sum} was given",
                )

        splitted_indices: dict = {}
        cluster_names = list(percentage_configuration.keys())
        classes_dictionary = DataSplit.generate_classes_dictionary(dataset)

        for cluster_size, cluster_name in zip(max_samples_per_cluster, cluster_names):
            for class_name, class_percentage in percentage_configuration[
                cluster_name
            ].items():
                split_size = int((class_percentage / 100) * cluster_size)
                class_indices = classes_dictionary[class_name]
                shuffle(class_indices)
                if split_size > len(class_indices):
                    raise ValueError(
                        f"The split size is greater than the \
                            number of samples in class {class_name}",
                    )
                if cluster_name in splitted_indices:
                    splitted_indices[cluster_name] += class_indices[:split_size]
                else:
                    splitted_indices[cluster_name] = class_indices[:split_size]
                classes_dictionary[class_name] = class_indices[split_size:]

        for name in cluster_names:
            splitted_indices[name] = np.array_split(
                np.array(splitted_indices[name]),
                num_nodes,
            )

        cluster_datasets, counter = DataSplit.split_cluster_dataset_in_parts(
            data=dataset,
            splitted_indices=splitted_indices,
            cluster_names=cluster_names,
        )

        return cluster_datasets, counter

    @staticmethod
    def percentage_sampling(
        dataset: torch.utils.data.DataLoader,
        percentage_configuration: dict,
        num_nodes: int = 1,
    ) -> tuple[list, list]:
        """This method is used to split the dataset using the percentage sampling.
        For each cluster we specify the percentage of samples we want in the cluster.
        For instance if we have 10 classes we could specify that we want 10% of each
        class in cluster 0. The percentage is specified as a dictionary where the key
        is the name of the cluster and the value is a dictionary with the percentage
        of each class. For instance:
            "percentage_configuration": {
                "cluster_0": {
                    "0": 10,
                    "1": 10,
                    "2": 10,
                    "3": 10,
                    "4": 10,
                    "5": 10,
                    "6": 10,
                    "7": 10,
                    "8": 10,
                    "9": 10
                },
            }.

        Example:


        Args:
            dataset (torch.utils.data.DataLoader): the dataset we want to split
            percentage_configuration (dict): the configuration of the
            cluster we want to create
            num_workers (int): total number of clusters we want to have

        Raises
        ------
            ValueError: When the configuration of the cluster is not correct
        """
        for cluster_name in percentage_configuration:
            tmp_sum = 0
            for _, value in percentage_configuration[cluster_name].items():
                if value > 100:
                    raise ValueError(
                        f"The percentage of each class must be less \
                            than 100, but {value} was given",
                    )
                tmp_sum += value
            if tmp_sum != 100:
                raise ValueError(
                    f"The sum of the percentages of each class must \
                         be equal to 100, but {tmp_sum} was given",
                )

        splitted_indices: dict = {}
        cluster_names = list(percentage_configuration.keys())
        classes_dictionary = DataSplit.generate_classes_dictionary(dataset)
        classes_dictionary_size = {
            class_name: len(class_indices)
            for class_name, class_indices in classes_dictionary.items()
        }

        for cluster_name in cluster_names:
            for class_name, class_percentage in percentage_configuration[
                cluster_name
            ].items():
                class_indices = classes_dictionary[class_name]
                split_size = int(
                    (class_percentage / 100) * classes_dictionary_size[class_name],
                )
                shuffle(class_indices)
                if split_size > len(class_indices):
                    raise ValueError(
                        f"The split size is greater than the \
                             number of samples in class {class_name}",
                    )
                if cluster_name in splitted_indices:
                    splitted_indices[cluster_name] += class_indices[:split_size]
                else:
                    splitted_indices[cluster_name] = class_indices[:split_size]
                classes_dictionary[class_name] = class_indices[split_size:]

        for name in cluster_names:
            splitted_indices[name] = np.array_split(
                np.array(splitted_indices[name]),
                num_nodes,
            )

        cluster_datasets, counter = DataSplit.split_cluster_dataset_in_parts(
            data=dataset,
            splitted_indices=splitted_indices,
            cluster_names=cluster_names,
        )

        return cluster_datasets, counter
