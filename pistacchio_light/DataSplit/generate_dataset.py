"""This file is used to generate the dataset for the experiments."""
# # Libraries import
import argparse, json, os, sys
# Modules import
from collections import Counter
from loguru import logger
# Cross-library imports
from pistacchio_light.DataSplit.data_split import DataSplit
from pistacchio_light.DataSplit.dataset_downloader import DatasetDownloader
from pistacchio_light.DataSplit.storage_manager import StorageManager
from pistacchio_light.Exceptions.errors import InvalidSplitTypeError
from pistacchio_simulator.Utils.preferences import Preferences

def print_debug(counters: Counter) -> None:
    """This prints the stats of the dataset.

    Args:
        counters (_type_): The stats of the dataset
    """
    classes = set()
    for counter in counters:
        keys = counter.keys()
        value_sum = sum(counter.values())
        for key in keys:
            classes.add(key)
        logger.info(
            f"{len(counter)}, {value_sum}, {sorted(counter.items(), key=lambda item: (item[0]))}"
        )
    logger.info("-----------------------")


def convert_targets_to_int(percentage_configuration: dict, targets: list) -> None:
    """Converts the targets that we specified in the percentage_configuration
    from str to int when the targets of the dataset are integers.

    Args:
        percentage_configuration (dict): the percentage configuration specified by the user
        targets (list): target list of the dataset we want to split
    """
    if percentage_configuration and (
        isinstance(targets[0], int) or isinstance(targets[0].item(), int)
    ):
        for key, value in percentage_configuration.items():
            percentage_configuration[key] = {int(k): v for k, v in value.items()}


def get_dataset(config: Preferences, custom_dataset: dict = None):
    """This function is used to download the dataset based on the
    preferences that we pass as parameter.

    Args:
        config (Preferences): preferences of the experiment

    Returns
    -------
        Tuple[torch.Dataset, torch.Dataset]: _description_
    """
    if custom_dataset:
        train_ds, test_ds = custom_dataset["train"], custom_dataset["test"]
    else:
        train_ds, test_ds = DatasetDownloader.download_dataset(config.dataset_name)
    return train_ds, test_ds


def pre_process_dataset(
    train_ds,
    percentage_underrepresented_classes,
    underrepresented_classes,
    num_reduced_nodes,
    num_samples_underrepresented_classes,
) -> None:
    """_summary_.

    Args:
        train_ds (_type_): _description_
        test_ds (_type_): _description_
        percentage_underrepresented_classes (_type_): _description_
        underrepresented_classes (_type_): _description_
        num_reduced_nodes (_type_): _description_
        num_samples_underrepresented_classes (_type_): _description_
    """
    if (
        percentage_underrepresented_classes
        and underrepresented_classes
        and not num_reduced_nodes
    ):
        DataSplit.reduce_samples(
            train_ds,
            underrepresented_classes,
            percentage_underrepresented_classes=percentage_underrepresented_classes,
        )
    elif (
        num_samples_underrepresented_classes
        and underrepresented_classes
        and not num_reduced_nodes
    ):
        DataSplit.reduce_samples(
            train_ds,
            underrepresented_classes,
            num_samples_underrepresented_classes=num_samples_underrepresented_classes,
        )


def stratified(train_ds, test_ds, num_clusters, num_nodes, max_samples_per_cluster):
    """_summary_.

    Args:
        train_ds (_type_): _description_
        test_ds (_type_): _description_
        num_clusters (_type_): _description_
        num_nodes (_type_): _description_
        max_samples_per_cluster (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    cluster_datasets, counters = DataSplit.stratified_sampling(
        dataset=train_ds,
        num_workers=num_clusters * num_nodes,
        max_samples_per_cluster=max_samples_per_cluster,
    )
    cluster_datasets_test, counters_test = DataSplit.stratified_sampling(
        dataset=test_ds,
        num_workers=num_clusters * num_nodes,
    )
    return cluster_datasets, counters, cluster_datasets_test, counters_test


def stratified_with_some_reduced(
    train_ds,
    test_ds,
    num_clusters,
    num_nodes,
    num_reduced_nodes,
    max_samples_per_cluster,
    underrepresented_classes,
    percentage_underrepresented_classes,
):
    """_summary_.

    Args:
        train_ds (_type_): _description_
        test_ds (_type_): _description_
        num_clusters (_type_): _description_
        num_nodes (_type_): _description_
        num_reduced_nodes (_type_): _description_
        max_samples_per_cluster (_type_): _description_
        underrepresented_classes (_type_): _description_
        percentage_underrepresented_classes (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    cluster_datasets, counters = DataSplit.stratified_sampling_with_some_nodes_reduced(
        dataset=train_ds,
        num_workers=num_clusters * num_nodes,
        num_reduced_nodes=num_reduced_nodes,
        max_samples_per_cluster=max_samples_per_cluster,
        underrepresented_classes=underrepresented_classes,
        percentage_underrepresented_classes=percentage_underrepresented_classes,
    )
    cluster_datasets_test, counters_test = DataSplit.stratified_sampling(
        dataset=test_ds,
        num_workers=num_clusters * num_nodes,
    )
    return cluster_datasets, counters, cluster_datasets_test, counters_test


def percentage_max_samples(
    config,
    train_ds,
    test_ds,
    num_clusters,
    max_samples_per_cluster,
):
    """_summary_.

    Args:
        config (_type_): _description_
        train_ds (_type_): _description_
        test_ds (_type_): _description_
        num_clusters (_type_): _description_
        max_samples_per_cluster (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    percentage_configuration_clusters = config.data_split_config[
        "percentage_configuration"
    ]
    percentage_configuration_clusters_test = config.data_split_config.get(
        "percentage_configuration_test",
        None,
    )

    convert_targets_to_int(
        percentage_configuration=percentage_configuration_clusters,
        targets=train_ds.targets,
    )
    convert_targets_to_int(
        percentage_configuration=percentage_configuration_clusters_test,
        targets=test_ds.targets,
    )

    cluster_datasets, counters = DataSplit.percentage_sampling_max_samples(
        dataset=train_ds,
        num_workers=num_clusters,
        max_samples_per_cluster=max_samples_per_cluster,
        percentage_configuration=percentage_configuration_clusters,
    )
    cluster_datasets_test, counters_test = DataSplit.percentage_sampling(
        dataset=train_ds,
        percentage_configuration=percentage_configuration_clusters,
    )
    return cluster_datasets, counters, cluster_datasets_test, counters_test


def percentage(config, train_ds, test_ds, num_nodes, task, names):
    """_summary_.

    Args:
        config (_type_): _description_
        train_ds (_type_): _description_
        test_ds (_type_): _description_
        num_nodes (_type_): _description_
        task (_type_): _description_
        names (_type_): _description_
    """
    percentage_configuration_clusters = config.data_split_config[
        "percentage_configuration"
    ]
    percentage_configuration_clusters_test = config.data_split_config.get(
        "percentage_configuration_test",
        None,
    )

    convert_targets_to_int(
        percentage_configuration=percentage_configuration_clusters,
        targets=train_ds.targets,
    )
    convert_targets_to_int(
        percentage_configuration=percentage_configuration_clusters_test,
        targets=test_ds.targets,
    )

    if config.data_split_config["noniid_nodes_distribution"]:
        iteration = 0
        nodes_distributions = {}
        # we consider each cluster and then we create a distribution for each node
        # of each cluster.
        for (
            cluster_name,
            cluster_distribution,
        ) in percentage_configuration_clusters.items():
            classes = list(cluster_distribution.keys())
            nodes_distributions_tmp = DataSplit.generate_nodes_distribution(
                num_nodes=num_nodes,
                classes=classes,
                names=names[iteration * num_nodes : iteration * num_nodes + num_nodes],
            )
            nodes_distributions[cluster_name] = nodes_distributions_tmp
            iteration += 1

        cluster_datasets, counters = DataSplit.percentage_split(
            dataset=train_ds,
            percentage_configuration=percentage_configuration_clusters,
            nodes_distribution=nodes_distributions,
            task=task,
        )
        cluster_datasets_test, counters_test = DataSplit.percentage_split(
            dataset=test_ds,
            percentage_configuration=percentage_configuration_clusters
            if percentage_configuration_clusters_test is None
            else percentage_configuration_clusters_test,
            nodes_distribution=nodes_distributions,
            task=task,
        )
    else:
        cluster_datasets, counters = DataSplit.percentage_split(
            dataset=train_ds,
            percentage_configuration=percentage_configuration_clusters,
            num_workers=num_nodes,
            task=task,
        )
        cluster_datasets_test, counters_test = DataSplit.percentage_split(
            dataset=test_ds,
            percentage_configuration=percentage_configuration_clusters
            if percentage_configuration_clusters_test is None
            else percentage_configuration_clusters_test,
            num_workers=num_nodes,
            task=task,
        )
    return cluster_datasets, counters, cluster_datasets_test, counters_test


def store_on_disk(config, cluster_datasets, cluster_datasets_test, test_ds, names):
    """_summary_.

    Args:
        config (_type_): _description_
        cluster_datasets (_type_): _description_
        cluster_datasets_test (_type_): _description_
        test_ds (_type_): _description_
        names (_type_): _description_
    """
    StorageManager.write_splitted_dataset(
        dataset_name=config.dataset_name,
        splitted_dataset=cluster_datasets,
        dataset_type="train_set",
        names=names,
    )

    StorageManager.write_validation_dataset(
        dataset_name=config.dataset_name,
        dataset=test_ds,
        dataset_type="server_validation",
    )
    StorageManager.write_splitted_dataset(
        dataset_name=config.dataset_name,
        splitted_dataset=cluster_datasets_test,
        dataset_type="test_set",
        names=names,
    )


def generate_splitted_dataset(config: Preferences, custom_dataset: dict = None) -> None:
    """This function is used to generate the dataset based on the
    configuration file passed as parameter.

    Args:
        config (_type_): configuration file

    Raises
    ------
        InvalidSplitTypeError: If we select a split type that is not valid
    """
    split_type = config.data_split_config["split_type"]
    underrepresented_classes = config.data_split_config.get(
        "underrepresented_class",
        None,
    )
    percentage_underrepresented_classes = config.data_split_config.get(
        "percentage_underrepresented_class",
        None,
    )
    num_samples_underrepresented_classes = config.data_split_config.get(
        "num_samples_underrepresented_classes",
        None,
    )
    max_samples_per_cluster = config.data_split_config.get(
        "max_samples_per_cluster",
        None,
    )
    num_reduced_nodes = config.data_split_config.get("num_reduced_nodes", None)

    train_ds, test_ds = get_dataset(config=config, custom_dataset=custom_dataset)
    pre_process_dataset(
        train_ds=train_ds,
        percentage_underrepresented_classes=percentage_underrepresented_classes,
        underrepresented_classes=underrepresented_classes,
        num_reduced_nodes=num_reduced_nodes,
        num_samples_underrepresented_classes=num_samples_underrepresented_classes,
    )

    task = config.task
    num_nodes = config.data_split_config["num_nodes"]
    num_clusters = config.data_split_config["num_clusters"]
    names = [
        f"{node_id}_cluster_{cluster_id}"
        for cluster_id in range(num_clusters)
        for node_id in range(num_nodes)
    ]

    if split_type == "stratified":
        cluster_datasets, counters, cluster_datasets_test, counters_test = stratified(
            train_ds=train_ds,
            test_ds=test_ds,
            num_clusters=num_clusters,
            num_nodes=num_nodes,
            max_samples_per_cluster=max_samples_per_cluster,
        )
    elif split_type == "stratified_with_some_reduced":
        (
            cluster_datasets,
            counters,
            cluster_datasets_test,
            counters_test,
        ) = stratified_with_some_reduced(
            train_ds=train_ds,
            test_ds=test_ds,
            num_clusters=num_clusters,
            num_nodes=num_nodes,
            num_reduced_nodes=num_reduced_nodes,
            max_samples_per_cluster=max_samples_per_cluster,
            underrepresented_classes=underrepresented_classes,
            percentage_underrepresented_classes=percentage_underrepresented_classes,
        )
    elif split_type == "percentage_max_samples":
        (
            cluster_datasets,
            counters,
            cluster_datasets_test,
            counters_test,
        ) = percentage_max_samples(
            config,
            train_ds,
            test_ds,
            num_clusters,
            max_samples_per_cluster,
        )
    elif split_type == "percentage":
        cluster_datasets, counters, cluster_datasets_test, counters_test = percentage(
            config,
            train_ds,
            test_ds,
            num_nodes,
            task,
            names,
        )
    else:
        raise InvalidSplitTypeError
    print_debug(counters)
    print_debug(counters_test)
    store_on_disk(config, cluster_datasets, cluster_datasets_test, test_ds, names)

class Dataset_Manager:
    def __init__(self) -> None:
        """Data storage is a class that can be used to download, transfer and save to disk
        data that can be used during the experiments with federated learning.
        Similarly to other classes in pistacchio_light, it can accept either preferences
        object (retrieved from json) or configuration dictionary"""
    
    def configure_dataset(self, json_settup):
        path = os.path.join("DatasetConfigurations", json_settup)
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)
            config = Preferences.generate_from_json(config)
            generate_splitted_dataset(config=config)