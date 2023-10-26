import os
import shutil
from collections import Counter

import numpy as np
import torch

from pistacchio_simulator.FederatedDataset.PartitionTypes.iid_partition import (
    IIDPartition,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.majority_minority_partition import (
    MajorityMinorityPartition,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.non_iid_partition import (
    NonIIDPartition,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.non_iid_partition_nodes import (
    NonIIDPartitionNodes,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.non_iid_partition_nodes_with_sensitive_feature import (
    NonIIDPartitionNodesWithSensitiveFeature,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.non_iid_partition_public_private import (
    NonIIDPartitionPublicPrivate,
)
from pistacchio_simulator.FederatedDataset.PartitionTypes.non_iid_partition_with_sensitive_feature import (
    NonIIDPartitionWithSensitiveFeature,
)
from pistacchio_simulator.FederatedDataset.Utils.custom_dataset import (
    MyDataset,
    MyDatasetWithCSV,
)
from pistacchio_simulator.FederatedDataset.Utils.dataset_downloader import (
    DatasetDownloader,
)
from pistacchio_simulator.FederatedDataset.Utils.preferences import Preferences


class FederatedDataset:
    def generate_partitioned_dataset(
        config: Preferences = None,
        split_type_clusters: str = None,
        split_type_nodes: str = None,
        num_nodes: int = None,
        num_clusters: int = None,
        num_classes: int = None,
        alpha: float = None,
        dataset_name: str = None,
        custom_dataset: dict = None,
        store_path: str = None,
        train_ds: torch.utils.data.Dataset = None,
        test_ds: torch.utils.data.Dataset = None,
        max_size: float = None,
        validation_size: float = None,
    ) -> None:
        dataset_with_csv = False
        if hasattr(train_ds, "image_path"):
            dataset_with_csv = True
        split_type_clusters = (
            config.data_split_config.split_type_clusters
            if config
            else split_type_clusters
        )
        split_type_nodes = (
            config.data_split_config.split_type_nodes if config else split_type_nodes
        )
        validation_size = (
            config.data_split_config.validation_size if config else validation_size
        )
        max_size = config.data_split_config.max_size if config else max_size
        num_nodes = config.data_split_config.num_nodes if config else num_nodes
        num_clusters = config.data_split_config.num_clusters if config else num_clusters
        num_classes = config.data_split_config.num_classes if config else num_classes
        alpha = config.data_split_config.alpha if config else alpha
        dataset_name = config.dataset if config else dataset_name
        store_path = config.data_split_config.store_path if config else store_path

        if train_ds is None and test_ds is None:
            train_ds, test_ds = DatasetDownloader.download_dataset(
                dataset_name=dataset_name,
            )
        data = train_ds.data if hasattr(train_ds, "data") else train_ds.samples
        cluster_splits_train = []
        cluster_splits_test = []
        cluster_splits_validation = []

        if os.path.exists(store_path):
            shutil.rmtree(store_path)
        os.makedirs(store_path)

        # At the moment we only consider two cases. The first one is the case in which
        # we want to split the dataset both among the clusters and among the nodes.
        # This is used for the experiments for the Federated learning with P2P.
        # In particular, at the moment the only split type supported for this case is
        # the majority minority in the cluster and the non iid among the nodes.
        # Other combinations of splits could work but I'm not sure about the results.
        # The other case is the one in which we only split the dataset among
        # the nodes. This is a classic Federated learning scenario.
        if num_nodes and num_clusters:
            # First we split the data among the clusters
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = FederatedDataset.partition_data(
                data=train_ds,
                split_type=split_type_clusters,
                num_partitions=num_clusters,
                alpha=alpha,
                num_classes=num_classes,
                phase="cluster",
                max_size=max_size,
            )
            (
                splitted_indexes_test,
                labels_per_cluster_test,
                samples_per_cluster_test,
            ) = FederatedDataset.partition_data(
                data=test_ds,
                split_type=split_type_clusters,
                num_partitions=num_clusters,
                alpha=alpha,
                num_classes=num_classes,
                phase="cluster",
                max_size=max_size,
            )

            for cluster_name, indexes in splitted_indexes_test.items():
                validation_test = IIDPartition.do_iid_partitioning_with_indexes(
                    indexes=np.array(test_ds.targets)[indexes], num_partitions=2
                )
                validation_indexes = validation_test[0]
                test_indexes = validation_test[1]
                if dataset_with_csv:
                    test_partition_cluster_validation = MyDatasetWithCSV(
                        targets=np.array(test_ds.targets)[validation_indexes],
                        image_path=test_ds.image_path,
                        image_ids=np.array(test_ds.samples)[validation_indexes],
                        transform=test_ds.transform,
                        sensitive_features=np.array(test_ds.sensitive_features)[
                            validation_indexes
                        ]
                        if hasattr(test_ds, "sensitive_features")
                        else None,
                    )
                    test_partition_cluster_test = MyDatasetWithCSV(
                        targets=np.array(test_ds.targets)[test_indexes],
                        image_path=test_ds.image_path,
                        image_ids=np.array(test_ds.samples)[test_indexes],
                        transform=test_ds.transform,
                        sensitive_features=np.array(test_ds.sensitive_features)[
                            test_indexes
                        ]
                        if hasattr(test_ds, "sensitive_features")
                        else None,
                    )
                else:
                    test_partition_cluster_validation = MyDataset(
                        targets=test_ds.targets[validation_indexes],
                        samples=np.array(test_ds.data)[validation_indexes],
                        transform=test_ds.transform,
                    )
                    test_partition_cluster_test = MyDataset(
                        targets=test_ds.targets[test_indexes],
                        samples=np.array(test_ds.data)[test_indexes],
                        transform=test_ds.transform,
                    )

                torch.save(
                    test_partition_cluster_test,
                    f"{store_path}/test_{cluster_name}.pt",
                )
                torch.save(
                    test_partition_cluster_validation,
                    f"{store_path}/validation_{cluster_name}.pt",
                )

            # And then we split the data among the nodes
            for cluster_name in splitted_indexes_train:
                current_labels_train = labels_per_cluster_train[cluster_name]
                current_samples_train = samples_per_cluster_train[cluster_name]
                current_labels_test = labels_per_cluster_test[cluster_name]
                current_samples_test = samples_per_cluster_test[cluster_name]

                # Create the training set for each node
                (
                    splitted_indexes_train_nodes,
                    labels_per_cluster_train_nodes,
                    samples_per_cluster_train_nodes,
                ) = FederatedDataset.partition_data(
                    data=MyDataset(
                        samples=current_samples_train,
                        targets=current_labels_train,
                        transform=train_ds.transform,
                    ),
                    split_type=split_type_nodes,
                    num_partitions=num_nodes,
                    alpha=alpha,
                    num_classes=num_classes,
                    partition_type="train",
                    phase="node",
                    max_size=max_size,
                )

                # splitted_indexes_train_nodes is a dictionary with the indexes
                # correponding to each node. labels per cluster train nodes is a
                # dictionary with the labels corresponding to each node.
                # Now we want to split the indexes and the corresponding labels
                # of each node into two parts: training and validation.
                # So we sample from the list of indexes and labels of each node
                # and we create the training and validation set for each node.

                validation_indexes = {}
                train_indexes = {}
                validation_labels = {}
                train_labels = {}

                for node_name in splitted_indexes_train_nodes:
                    indexes = splitted_indexes_train_nodes[node_name]
                    labels = labels_per_cluster_train_nodes[node_name]
                    # sample from indexes list the validation_size * len(indexes) indexes
                    # using the random sample function
                    # and use them as validation set. Remove them from indexes.
                    # The remaining indexes will be used as training set.
                    sampled_validation_indexes = np.random.choice(
                        range(len(indexes)),
                        int(validation_size * len(indexes)),
                        replace=False,
                    )
                    # get the remaining indexes
                    sampled_train_indexes = np.array(
                        list(set(range(len(indexes))) - set(sampled_validation_indexes))
                    )

                    validation_indexes[node_name] = indexes[sampled_validation_indexes]
                    train_indexes[node_name] = indexes
                    validation_labels[node_name] = np.array(labels)[
                        sampled_validation_indexes
                    ]
                    train_labels[node_name] = labels
                    print(
                        f"Node {node_name} - Cluster {cluster_name} has {len(indexes)} samples - {len(sampled_validation_indexes)} Validation Set - {len(sampled_train_indexes)} Train Set"
                    )

                cluster_splits_train.append(
                    (
                        cluster_name,
                        splitted_indexes_train_nodes,
                        labels_per_cluster_train_nodes,
                    ),
                )

                cluster_splits_validation.append(
                    (
                        cluster_name,
                        validation_indexes,
                        validation_labels,
                    ),
                )

                # Create the test set for each node
                (
                    splitted_indexes_test_nodes,
                    labels_per_cluster_test_nodes,
                    samples_per_cluster_test_nodes,
                ) = FederatedDataset.partition_data(
                    data=MyDataset(
                        samples=current_samples_test,
                        targets=current_labels_test,
                        transform=train_ds.transform,
                    ),
                    split_type=split_type_nodes,
                    num_partitions=num_nodes,
                    alpha=alpha,
                    num_classes=num_classes,
                    phase="node",
                    max_size=max_size,
                )

                cluster_splits_test.append(
                    (
                        cluster_name,
                        splitted_indexes_test_nodes,
                        labels_per_cluster_test_nodes,
                    )
                )
                # At the end of this loop we have a list of tuples. Each tuple contains
                # The name of the clusters, the indexes of the samples for each node
                # and the labels for each node.

        elif num_nodes:
            # In this second case we only want to partition the dataset among
            # the nodes.
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = FederatedDataset.partition_data(
                data=train_ds,
                split_type=split_type_nodes,
                num_partitions=num_nodes,
                alpha=alpha,
                num_classes=num_classes,
                phase="node",
                max_size=max_size,
            )
            (
                splitted_indexes_test,
                labels_per_cluster_test,
                samples_per_cluster_test,
            ) = FederatedDataset.partition_data(
                data=test_ds,
                split_type=split_type_nodes,
                num_partitions=num_nodes,
                alpha=alpha,
                num_classes=num_classes,
                phase="node",
                max_size=max_size,
            )

        # We take some information from the dataset. We need these information
        # because then we will create new partitioned datasets that will
        # contain these informations.
        dataset_with_csv = False
        if hasattr(train_ds, "image_path"):
            image_path_train = train_ds.image_path
            samples_train = train_ds.samples
            image_path_test = test_ds.image_path
            samples_test = test_ds.samples
            dataset_with_csv = True
        else:
            image_path_train = None
            image_path_test = None

        if hasattr(train_ds, "data"):
            samples_train = train_ds.data
            samples_test = test_ds.data
        transform_train = train_ds.transform
        transform_test = train_ds.transform

        targets_train_per_class = FederatedDataset.create_targets_per_class(
            data=train_ds,
        )
        targets_test_per_class = FederatedDataset.create_targets_per_class(
            data=test_ds,
        )
        targets_validation_per_class = FederatedDataset.create_targets_per_class(
            data=train_ds,
        )

        partitions_train = None
        partitions_test = None
        # Finally, we create the splitted datasets
        # There are two cases: the one in which we splitted both among the
        # clusters and among the nodes and the one in which we splitted only
        # among the nodes
        if cluster_splits_train:
            partitions_train = (
                FederatedDataset.create_partitioned_dataset_with_clusters(
                    cluster_splits=cluster_splits_train,
                    targets_per_class=targets_train_per_class,
                    dataset_with_csv=dataset_with_csv,
                    dataset=train_ds,
                    image_path=image_path_train,
                    transform=transform_train,
                )
            )

            partitions_test = FederatedDataset.create_partitioned_dataset_with_clusters(
                cluster_splits=cluster_splits_test,
                targets_per_class=targets_test_per_class,
                dataset_with_csv=dataset_with_csv,
                dataset=test_ds,
                image_path=image_path_test,
                transform=transform_test,
            )

            merged_split_validation = []
            for cluster_distribution in cluster_splits_validation:
                cluster_name = cluster_distribution[0]
                tmp_dict = {}
                tmp_dict_labels = {}
                for node_name, node_indexes in cluster_distribution[1].items():
                    if "_private" in node_name:
                        new_name = node_name.split("_private")[0]
                    else:
                        new_name = node_name.split("_public")[0]
                    if new_name not in tmp_dict:
                        tmp_dict[new_name] = []
                    tmp_dict[new_name] += list(node_indexes)
                for node_name, node_labels in cluster_distribution[2].items():
                    if "_private" in node_name:
                        new_name = node_name.split("_private")[0]
                    else:
                        new_name = node_name.split("_public")[0]
                    if new_name not in tmp_dict_labels:
                        tmp_dict_labels[new_name] = []
                    tmp_dict_labels[new_name] += list(node_labels)

                merged_split_validation.append(
                    (cluster_name, tmp_dict, tmp_dict_labels),
                )

            partitions_validation = (
                FederatedDataset.create_partitioned_dataset_with_clusters(
                    cluster_splits=merged_split_validation,
                    targets_per_class=targets_validation_per_class,
                    dataset_with_csv=dataset_with_csv,
                    dataset=train_ds,
                    image_path=image_path_test,
                    transform=transform_test,
                    validation=True,
                )
            )

        else:
            partitions_train = FederatedDataset.create_partitioned_dataset(
                labels_per_cluster=labels_per_cluster_train,
                targets_per_class=targets_train_per_class,
                dataset_with_csv=dataset_with_csv,
                dataset=train_ds,
                image_path=image_path_train,
                transform=transform_train,
            )
            partitions_test = FederatedDataset.create_partitioned_dataset(
                labels_per_cluster=labels_per_cluster_test,
                targets_per_class=targets_test_per_class,
                dataset_with_csv=dataset_with_csv,
                dataset=test_ds,
                image_path=image_path_test,
                transform=transform_train,
            )

        # Now we can store the partitioned datasets
        FederatedDataset.store_partitioned_datasets(
            partitions_train,
            store_path=store_path,
            split_name="train",
        )
        FederatedDataset.store_partitioned_datasets(
            partitions_test,
            store_path=store_path,
            split_name="test",
        )

        FederatedDataset.store_partitioned_datasets(
            partitions_validation,
            store_path=store_path,
            split_name="validation",
        )

        validation_test = IIDPartition.do_iid_partitioning_with_indexes(
            indexes=torch.tensor(range(len(test_ds.targets))), num_partitions=2
        )
        validation_indexes = validation_test[0]
        test_indexes = validation_test[1]

        if dataset_with_csv:
            validation_partition = MyDatasetWithCSV(
                targets=np.array(test_ds.targets)[validation_indexes]
                if isinstance(test_ds.targets, list)
                else test_ds.targets[validation_indexes],
                image_path=image_path_test,
                image_ids=np.array(test_ds.samples)[validation_indexes],
                transform=test_ds.transform,
                sensitive_features=torch.tensor(test_ds.sensitive_features)[
                    validation_indexes
                ]
                if hasattr(test_ds, "sensitive_features")
                else None,
            )
        else:
            validation_partition = MyDataset(
                targets=test_ds.targets[validation_indexes],
                samples=np.array(test_ds.data)[validation_indexes],
                transform=train_ds.transform,
            )

        FederatedDataset.store_validation_set(
            validation_partition,
            store_path=store_path,
            split_name="server_validation_set",
        )

        if dataset_with_csv:
            test_partition = MyDatasetWithCSV(
                targets=np.array(test_ds.targets)[test_indexes]
                if isinstance(test_ds.targets, list)
                else test_ds.targets[test_indexes],
                image_path=image_path_test,
                image_ids=np.array(test_ds.samples)[test_indexes],
                transform=test_ds.transform,
                sensitive_features=torch.tensor(test_ds.sensitive_features)[
                    test_indexes
                ]
                if hasattr(test_ds, "sensitive_features")
                else None,
            )
        else:
            test_partition = MyDataset(
                targets=test_ds.targets[test_indexes],
                samples=np.array(test_ds.data)[test_indexes],
                transform=train_ds.transform,
            )

        FederatedDataset.store_validation_set(
            test_partition,
            store_path=store_path,
            split_name="server_test_set",
        )

    def create_targets_per_class(data):
        """This function creates a dictionary containing the targets of
        the datasets as key and the indexes of the samples belonging to
        that class as values.

        Args:
            data (_type_): the dataset we are partitioning
        """
        targets_per_class = {}

        for index, target in enumerate(data.targets):
            target = target.item() if isinstance(target, torch.Tensor) else target
            if target not in targets_per_class:
                targets_per_class[target] = []
            targets_per_class[target].append(
                index.item() if isinstance(index, torch.Tensor) else index,
            )
        return targets_per_class

    def store_partitioned_datasets(
        partitioned_datasets: list,
        store_path: str,
        split_name: str,
    ):
        for index, partition_name in enumerate(partitioned_datasets):
            if isinstance(partitioned_datasets[partition_name], dict):
                for node_name in partitioned_datasets[partition_name]:
                    torch.save(
                        partitioned_datasets[partition_name][node_name],
                        f"{store_path}/{partition_name}_{node_name}_{split_name}.pt",
                    )
            else:
                # When we split the dataset only among the nodes we want to
                # store the partitioned dataset in folders. Each folder will
                # contain the partitioned dataset of a node. This is
                # useful when we want to use Flower to train the federated model.
                directory = f"{store_path}/{index}/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(
                    partitioned_datasets[partition_name],
                    f"{directory}{partition_name}_{split_name}.pt",
                )

    def store_validation_set(
        dataset,
        store_path: str,
        split_name: str,
    ):
        directory = f"{store_path}/"

        torch.save(
            dataset,
            f"{directory}{split_name}.pt",
        )

    def partition_data(
        data,
        split_type: str,
        num_partitions: int,
        alpha: float,
        num_classes: int,
        partition_type: str = None,
        phase: str = None,
        max_size: float = None,
    ):
        print(f"NUM CLASSES {num_classes}")
        samples_per_cluster_train = []
        if split_type == "iid":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = IIDPartition.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
            )

        elif split_type == "majority_minority":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = MajorityMinorityPartition.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
            )

        elif split_type == "non_iid":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = NonIIDPartition.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
                phase=phase,
            )
        elif split_type == "non_iid_double":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = NonIIDPartition.do_partitioning_double(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
                phase=phase,
            )
        elif split_type == "non_iid_public_private":
            if partition_type == "train":
                (
                    splitted_indexes_train,
                    labels_per_cluster_train,
                ) = NonIIDPartitionPublicPrivate.do_partitioning(
                    dataset=data,
                    num_partitions=num_partitions,
                    total_num_classes=num_classes,
                    alpha=alpha,
                )
            else:
                (
                    splitted_indexes_train,
                    labels_per_cluster_train,
                    samples_per_cluster_train,
                ) = NonIIDPartition.do_partitioning(
                    dataset=data,
                    num_partitions=num_partitions,
                    total_num_classes=num_classes,
                    alpha=alpha,
                )
        elif split_type == "non_iid_public_private_different_distribution":
            if partition_type == "train":
                (
                    splitted_indexes_train,
                    labels_per_cluster_train,
                ) = NonIIDPartitionPublicPrivate.do_partitioning_different_distribution_nodes(
                    dataset=data,
                    num_partitions=num_partitions,
                    total_num_classes=num_classes,
                    alpha=alpha,
                    max_size=max_size,
                )
            else:
                (
                    splitted_indexes_train,
                    labels_per_cluster_train,
                    samples_per_cluster_train,
                ) = NonIIDPartition.do_partitioning(
                    dataset=data,
                    num_partitions=num_partitions,
                    total_num_classes=num_classes,
                    alpha=alpha,
                )
        elif split_type == "non_iid_nodes":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = NonIIDPartitionNodes.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
            )
        elif split_type == "non_iid_sensitive_feature":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = NonIIDPartitionWithSensitiveFeature.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
            )
        elif split_type == "non_iid_nodes_sensitive_feature":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = NonIIDPartitionNodesWithSensitiveFeature.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
            )
        return (
            splitted_indexes_train,
            labels_per_cluster_train,
            samples_per_cluster_train,
        )

    def create_partitioned_dataset(
        labels_per_cluster,
        targets_per_class,
        dataset_with_csv,
        dataset,
        image_path,
        transform,
    ):
        partitions = {}
        for partition_name, labels in labels_per_cluster.items():
            counter_labels = Counter(labels)
            indexes = []
            for label, count in counter_labels.items():
                if isinstance(label, torch.Tensor):
                    label = label.item()
                indexes += targets_per_class[label][:count]
                targets_per_class[label] = targets_per_class[label][count:]
            if dataset_with_csv:
                if isinstance(dataset.targets, list):
                    dataset.targets = torch.tensor(dataset.targets)
                train_partition = MyDatasetWithCSV(
                    targets=dataset.targets[indexes],
                    image_path=image_path,
                    image_ids=np.array(dataset.samples)[indexes],
                    transform=transform,
                    sensitive_features=np.array(dataset.sensitive_features)[indexes]
                    if hasattr(dataset, "sensitive_features")
                    else None,
                )
            else:
                train_partition = MyDataset(
                    targets=dataset.targets[indexes],
                    samples=torch.tensor(dataset.data)[indexes].to(torch.float32),
                    transform=train_ds.transform,
                )
            partitions[partition_name] = train_partition
        for partition_name, partition in partitions.items():
            if hasattr(partition, "sensitive_features"):
                print(
                    f"Partition {partition_name} has {len(partition)} samples, {len(set([item.item() for item in partition.sensitive_features]))} sensitive_features: {Counter([(target.item(), feature.item()) for target, feature in zip(partition.targets, partition.sensitive_features)])}",
                )
            else:
                print(
                    f"Partition {partition_name} has {len(partition)} samples: {Counter([item.item() for item in partition.targets])}",
                )
        print("_________________________")

        return partitions

    def create_partitioned_dataset_with_clusters(
        cluster_splits,
        targets_per_class,
        dataset_with_csv,
        dataset,
        image_path,
        transform,
        validation=False,
    ):
        if validation:
            print("------------------------------------")
        partitions = {}
        counter_partitions = {}
        for (
            cluster_name,
            _,
            splitted_labels_cluster,
        ) in cluster_splits:
            counter_partitions[cluster_name] = []
            partitions[cluster_name] = {}

            # Now we have the indexes of the data for each node
            # We can create the dataset for each node
            for node_name, labels in splitted_labels_cluster.items():
                counter_labels = Counter(labels)
                print(
                    f"Node {node_name} has {sum(counter_labels.values())} samples - distribution: {counter_labels}"
                )
                indexes = []
                for label, count in counter_labels.items():
                    label = label.item() if isinstance(label, torch.Tensor) else label
                    indexes += targets_per_class[label][:count]
                    targets_per_class[label] = targets_per_class[label][count:]
                if dataset_with_csv:
                    train_partition = MyDatasetWithCSV(
                        targets=np.array(dataset.targets)[indexes]
                        if isinstance(dataset.targets, list)
                        else dataset.targets[indexes],
                        image_path=image_path,
                        image_ids=np.array(dataset.samples)[indexes],
                        transform=transform,
                        sensitive_features=torch.tensor(dataset.sensitive_features)[
                            indexes
                        ]
                        if hasattr(dataset, "sensitive_features")
                        else None,
                    )
                    partitions[cluster_name][node_name] = train_partition
                else:
                    train_partition = MyDataset(
                        targets=dataset.targets[indexes],
                        samples=np.array(dataset.data)[indexes],
                        transform=transform,
                    )
                    partitions[cluster_name][node_name] = train_partition
                counter_partitions[cluster_name].append(counter_labels)

        for cluster_name in counter_partitions:
            print(
                f"Cluster {cluster_name} has {sum(counter_partitions[cluster_name], Counter())}",
            )
        print("_________________________/")
        return partitions
