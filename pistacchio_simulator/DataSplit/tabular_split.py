from pistacchio_simulator.DataSplit.data_split import DataSplit
from pistacchio_simulator.DataSplit.storage_manager import StorageManager
from pistacchio_simulator.Utils.preferences import Preferences
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from collections import Counter


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


class TabularSplit:
    @staticmethod
    def sample_random_indexes(config: Preferences, dataframe):
        allRows = dataframe.index.values
        num_clients = config.data_split_config["num_clusters"]
        samples = []
        num_samples = len(allRows) // num_clients

        for _ in range(num_clients):
            choices = np.random.choice(allRows, num_samples, replace=False)
            samples.append(choices)
            allRows = np.setdiff1d(allRows, choices)
        return samples

    @staticmethod
    def convert_images_to_index(dataset: Dataset) -> dict:
        """This function creates a dictionary where the keys
        are the names of the files that we have in the dataset
        and the values are the indexes in the dataset passed
        as parameter.

        Example:
        >>> convert_images_to_index(dataset=fair_face_dataset)
            {'9997.jpg': 86742, '9998.jpg': 86743}

        Args:
            dataset (Dataset): the training dataset we want to
                split. This is the dataset composed of images
                and not the csv file.
        Returns:
            Dict: a dictionary containing the file names with the
                corresponding indexes.
        """
        images = {}
        for index, (file_name, _) in enumerate(dataset.samples):
            file_name = file_name.split("/")[-1]
            images[file_name] = index
        return images

    @staticmethod
    def convert_indexes_to_path(
        config: Preferences, images: dict, dataframe: pd.DataFrame
    ) -> list:
        """This function generates a list. Each item in the list
        will be the index of the corresponding image in the dataset.
        For instance, if we are considering the image 1.jpg, we will
        insert in the returned list the index 105 because this image is
        indexes as 105 in the dataset.
        This is useful when we want to use the csv to filter data from the
        dataset because the index of the dataset and the index of the csv
        do not have a precise correspondence.

        Args:
            config (Preferences): The configuration we want to use in this
                experiment
            images (dict): a dictionary <path, index>
            dataframe (pd.Dataframe): the csv that contains all the additional
                information about the dataset
        """
        samples = TabularSplit.sample_random_indexes(config=config, dataframe=dataframe)
        index_lists = []
        for sample in samples:
            sampled_data = dataframe.loc[sample]
            paths = []
            for _, row in sampled_data.iterrows():
                paths.append(images[row["file"].split("/")[-1]])

            index_lists.append(paths)
        return index_lists

    @staticmethod
    def random_sample(config: Preferences, dataset: Dataset, dataframe: pd.DataFrame):
        """This function is used to generate random samples from the
        dataset passed as parameter.

        Args:
            config (Preferences): The configuration we want to use in our
                experiments
            dataset (Dataset): the dataset we want to split
        """
        images = TabularSplit.convert_images_to_index(dataset=dataset)
        index_lists = TabularSplit.convert_indexes_to_path(
            config=config, images=images, dataframe=dataframe
        )
        subsets = []
        for index_list in index_lists:
            subset = Subset(dataset, index_list)
            # _, targets = DataSplit.convert_subset_to_dataset(subset)
            # counters.append(Counter(targets))
            subsets.append(subset)

        return subsets

    @staticmethod
    def stratified_sampling():
        pass

    @staticmethod
    def split_dataset(config: Preferences, train_ds: Dataset, test_ds: Dataset):
        split_type = config.data_split_config["split_type"]
        num_nodes = config.data_split_config["num_nodes"]
        num_clusters = config.data_split_config["num_clusters"]
        names = [
            f"{node_id}_cluster_{cluster_id}"
            for cluster_id in range(num_clusters)
            for node_id in range(num_nodes)
        ]

        match split_type:
            case "random":
                subsets_train = TabularSplit.random_sample(
                    config=config,
                    dataset=train_ds,
                    dataframe=pd.read_csv(config.data_split_config["csv_train"]),
                )
                subsets_test = TabularSplit.random_sample(
                    config=config,
                    dataset=test_ds,
                    dataframe=pd.read_csv(config.data_split_config["csv_val"]),
                )
            case "stratified":
                TabularSplit.stratified_sampling()
            case _:
                print("OK")

        store_on_disk(config, subsets_train, subsets_test, test_ds, names)
