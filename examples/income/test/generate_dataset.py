import argparse
import json
import random

# matpllitb
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from pistacchio_simulator.FederatedDataset.partition_dataset import FederatedDataset
from pistacchio_simulator.FederatedDataset.Utils.custom_dataset import TabularDataset
from pistacchio_simulator.FederatedDataset.Utils.preferences import Preferences


class ACSIncomeSplit:
    def split_indexes_iid(num_clients, num_samples):
        indexes = np.arange(num_samples)
        np.random.shuffle(indexes)
        return np.array_split(indexes, num_clients)

    def split_indexes_non_iid(labels_and_grops_indexes, num_clients, alpha):
        labels = []
        groups = []
        indexes = {}
        for label, group, _ in labels_and_grops_indexes:
            labels.append(label)
            groups.append(group)

        label_and_group = list(zip(labels, groups))
        for index, (label, group) in enumerate(label_and_group):
            if (label, group) not in indexes:
                indexes[(label, group)] = []
            indexes[(label, group)].append(index)

        possible_labels_and_groups = list(set(label_and_group))
        # generate len(possible_labels_and_groups) dirichlet distributions with alpha
        # we need a distribution for each possible (label, group) and for each distribution
        # we need to have a probability for each client
        distributions = []
        for partition in range(len(possible_labels_and_groups)):
            distribution = np.random.dirichlet(num_clients * [alpha], size=1)
            distributions.append(distribution[0])

        # now we sample from each of the distributions to get the indexes for each client
        # we do not want to sample the same index twice
        clients_indexes = []
        for client in range(num_clients):
            client_indexes = []
            for partition in range(len(possible_labels_and_groups)):
                indexes_for_partition = indexes[possible_labels_and_groups[partition]]
                num_samples = int(
                    distributions[partition][client] * len(indexes_for_partition)
                )
                samples = np.random.choice(
                    indexes_for_partition, num_samples, replace=False
                )
                client_indexes.extend(samples)
            clients_indexes.append(client_indexes)

        return clients_indexes


# class TabularDataset(Dataset):
#     def __init__(self, x, z, y):
#         """
#         Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

#         Args:
#         x (list of tensors): List of input feature tensors.
#         z (list): List of sensitive values.
#         y (list): List of target values.
#         """
#         self.samples = x
#         self.sensitive_features = z
#         self.sensitive_attribute = z
#         self.gender = z
#         self.classes = y
#         self.targets = y

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         """
#         Get a single data point from the dataset.

#         Args:
#         idx (int): Index to retrieve the data point.

#         Returns:
#         sample (dict): A dictionary containing 'x', 'z', and 'y'.
#         """
#         x_sample = self.samples[idx]
#         z_sample = self.sensitive_features[idx]
#         y_sample = self.targets[idx]

#         return x_sample, z_sample, y_sample


parser = argparse.ArgumentParser(description="Experiments from config file")
parser.add_argument(
    "--config",
    type=str,
    default="",
    metavar="N",
    help="Config file",
)
args = parser.parse_args()
config = None
with open(args.config, "r", encoding="utf-8") as file:
    config = json.load(file)
config = Preferences(**config)

num_clusters = config.data_split_config.num_clusters
max_len_public = config.data_split_config.max_size
num_clients = config.data_split_config.num_nodes

test_size = 0.2
validation_size = config.data_split_config.validation_size
seed = config.data_split_config.seed
random.seed(seed)

public_private_indexes = []
public_private_indexes_validation = []

clusters_test_data = []

for cluster in range(num_clusters):
    dataframe = np.load(
        f"../data/income/federated/{cluster}/income_dataframes_{cluster}.npy"
    )
    income_groups = np.load(
        f"../data/income/federated/{cluster}/income_groups_{cluster}.npy"
    )
    income_labels = np.load(
        f"../data/income/federated/{cluster}/income_labels_{cluster}.npy"
    )

    # sample from a numpy array
    test_indexes = np.arange(len(dataframe))
    num_samples_test = int(len(dataframe) * test_size)
    sampled_indexes_test = np.random.choice(
        test_indexes, num_samples_test, replace=False
    )

    dataframe_test = dataframe[sampled_indexes_test]
    income_groups_test = income_groups[sampled_indexes_test]
    income_labels_test = income_labels[sampled_indexes_test]

    cluster_test_data = TabularDataset(
        dataframe_test, income_groups_test, income_labels_test
    )
    clusters_test_data.append((dataframe_test, income_groups_test, income_labels_test))
    torch.save(
        cluster_test_data, f"../data/income/federated_data/test_cluster_{cluster}.pt"
    )

    # remove the sampled_indexes_test from the original numpy array
    dataframe = np.delete(dataframe, sampled_indexes_test, axis=0)
    income_groups = np.delete(income_groups, sampled_indexes_test)
    income_labels = np.delete(income_labels, sampled_indexes_test)

    dataset_test = TabularDataset(
        dataframe_test, income_groups_test, income_labels_test
    )

    labels_and_groups_indexes = [
        (lab, group, index)
        for lab, group, index in zip(
            list(income_labels),
            list(income_groups),
            range(len(income_labels)),
        )
    ]

    client_indexes = ACSIncomeSplit.split_indexes_non_iid(
        labels_and_groups_indexes, num_clients=num_clients, alpha=5
    )
    labels_and_group_indexes_public_private = []
    # for each client we get the labels and the groups because we need to
    # create the public and private datasets
    for client in client_indexes:
        label = income_labels[client]
        group = income_groups[client]
        index = client
        labels_and_group_indexes_public_private.append(
            [(lab, group, ind) for lab, group, ind in zip(label, group, index)]
        )

    tmp_public_private_indexes = []
    tmp_public_private_indexes_validation = []
    for client_id, client in enumerate(labels_and_group_indexes_public_private):
        total_size_data = len(client)
        indexes = [index for (label, group, index) in client]
        public_size_data = int(total_size_data * max_len_public)
        private_size_data = total_size_data - public_size_data
        # random sample public_size_data samples from the indexes of the client
        # based on a dirichlet distribution
        distribution = np.random.dirichlet(private_size_data * [1], size=1)
        # private_indexes = np.random.choice(
        #     indexes, private_size_data, replace=False, p=distribution[0]
        # )
        private_indexes = np.random.choice(indexes, private_size_data, replace=False)
        private_indexes_validation = np.random.choice(
            private_indexes, int(private_size_data * validation_size), replace=False
        )
        private_indexes = [index for index in indexes if index not in private_indexes]

        private_data = dataframe[private_indexes]
        private_groups = income_groups[private_indexes]
        private_labels = income_labels[private_indexes]
        private_data_validation = dataframe[private_indexes_validation]
        private_groups_validation = income_groups[private_indexes_validation]
        private_labels_validation = income_labels[private_indexes_validation]

        private_dataset = TabularDataset(private_data, private_groups, private_labels)
        private_dataset_validation = TabularDataset(
            private_data_validation,
            private_groups_validation,
            private_labels_validation,
        )

        torch.save(
            private_dataset,
            f"../data/income/federated_data/cluster_{cluster}_node_{client_id}_private_train.pt",
        )
        torch.save(
            private_dataset_validation,
            f"../data/income/federated_data/cluster_{cluster}_node_{client_id}_private_validation.pt",
        )

        public_indexes = [index for index in indexes if index not in private_indexes]
        public_indexes_validation = np.random.choice(
            public_indexes, int(public_size_data * validation_size), replace=False
        )
        public_indexes = [index for index in indexes if index not in public_indexes]

        public_data = dataframe[public_indexes]
        public_groups = income_groups[public_indexes]
        public_labels = income_labels[public_indexes]
        public_data_validation = dataframe[public_indexes_validation]
        public_groups_validation = income_groups[public_indexes_validation]
        public_labels_validation = income_labels[public_indexes_validation]

        public_dataset = TabularDataset(public_data, public_groups, public_labels)
        public_dataset_validation = TabularDataset(
            public_data_validation, public_groups_validation, public_labels_validation
        )

        torch.save(
            public_dataset,
            f"../data/income/federated_data/cluster_{cluster}_node_{client_id}_public_train.pt",
        )
        torch.save(
            public_dataset_validation,
            f"../data/income/federated_data/cluster_{cluster}_node_{client_id}_public_validation.pt",
        )

        tmp_public_private_indexes.append((public_indexes, private_indexes))
        tmp_public_private_indexes_validation.append(
            (public_indexes_validation, private_indexes_validation)
        )

    public_private_indexes.append(tmp_public_private_indexes)
    public_private_indexes_validation.append(tmp_public_private_indexes_validation)


data = [data for (data, _, _) in clusters_test_data]
# concatenate the numpy arrays
data = np.concatenate(data, axis=0)
groups = [groups for (_, groups, _) in clusters_test_data]
groups = np.concatenate(groups, axis=0)
labels = [labels for (_, _, labels) in clusters_test_data]
labels = np.concatenate(labels, axis=0)

test_dataset = TabularDataset(data, groups, labels)
torch.save(test_dataset, "../data/income/federated_data/server_test_set.pt")



