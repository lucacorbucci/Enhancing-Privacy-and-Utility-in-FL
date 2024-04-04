import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from torchvision import datasets, transforms

from pistacchio_simulator.FederatedDataset.Utils.custom_dataset import (
    CelebaDataset,
    # CelebaGenderDataset,
    ImaginetteDataset,
    MyDataset,
    TabularDataset,
)


class DatasetDownloader:
    """This class is used to download some useful datasets.

    Raises
    ------
        InvalidDatasetName: If the dataset name is not valid
    """

    @staticmethod
    def download_dataset(dataset_name: str) -> Tuple:
        """Downloads the dataset with the given name.

        Args:
            dataset_name (str): The name of the dataset to download
        Raises:
            InvalidDatasetName: If the dataset name is not valid

        Returns
        -------
            _type_: The downloaded dataset
        """
        if dataset_name == "mnist":
            train_ds, test_ds = DatasetDownloader.download_mnist()
        elif dataset_name == "cifar10":
            train_ds, test_ds = DatasetDownloader.download_cifar10()
        elif dataset_name == "adult":
            train_ds, test_ds = DatasetDownloader.download_adult()
        elif dataset_name == "dutch":
            train_ds, test_ds = DatasetDownloader.download_dutch()
        elif dataset_name == "celeba":
            train_ds, test_ds = DatasetDownloader.download_celeba()
        # elif dataset_name == "celeba_sensitive_feature":
        #     train_ds, test_ds = DatasetDownloader.download_celeba_sensitive_feature()
        # elif dataset_name == "celeba_gender":
        #     train_ds, test_ds = DatasetDownloader.download_celeba_gender()
        elif dataset_name == "fashion_mnist":
            train_ds, test_ds = DatasetDownloader.download_fashion_mnist()
        elif dataset_name == "imaginette":
            train_ds, test_ds = DatasetDownloader.download_imaginette()
        elif dataset_name == "imaginette_csv":
            train_ds, test_ds = DatasetDownloader.download_imaginette_with_csv()
        else:
            raise ValueError("Invalid dataset name")
        return train_ds, test_ds

    @staticmethod
    def download_mnist() -> (
        Tuple[
            torchvision.datasets.MNIST,
            torchvision.datasets.MNIST,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ],
        )
        mnist_train_ds = torchvision.datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )

        mnist_test_ds = torchvision.datasets.MNIST(
            "../data",
            train=False,
            transform=transform,
        )
        return mnist_train_ds, mnist_test_ds

    @staticmethod
    def download_fashion_mnist() -> (
        Tuple[
            torchvision.datasets.FashionMNIST,
            torchvision.datasets.FashionMNIST,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        )
        mnist_train_ds = torchvision.datasets.FashionMNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )

        mnist_test_ds = torchvision.datasets.FashionMNIST(
            "../data",
            train=False,
            transform=transform,
        )
        return mnist_train_ds, mnist_test_ds

    @staticmethod
    def download_imaginette() -> (
        Tuple[
            torchvision.datasets.ImageFolder,
            torchvision.datasets.ImageFolder,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

        mnist_train_ds = torchvision.datasets.ImageFolder(
            "../data/imaginette/train",
            transform=transform,
        )

        mnist_test_ds = torchvision.datasets.ImageFolder(
            "../data/imaginette/val",
            transform=transform,
        )
        return mnist_train_ds, mnist_test_ds

    @staticmethod
    def download_imaginette_with_csv() -> (
        Tuple[
            torchvision.datasets.ImageFolder,
            torchvision.datasets.ImageFolder,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

        imaginette_train_ds = ImaginetteDataset(
            csv_path="../data/imaginette_csv/train.csv",
            image_path="../data/imaginette_csv/train",
            transform=transform,
        )

        imaginette_test_ds = ImaginetteDataset(
            csv_path="../data/imaginette_csv/test.csv",
            image_path="../data/imaginette_csv/val",
            transform=transform,
        )
        return imaginette_train_ds, imaginette_test_ds

    @staticmethod
    def download_celeba() -> (
        Tuple[
            torchvision.datasets.CelebA,
            torchvision.datasets.CelebA,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )
        print("Downloading Celeba")
        train_dataset = CelebaDataset(
            csv_path="../data/celeba/train_original.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        test_dataset = CelebaDataset(
            csv_path="../data/celeba/test_original.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        return train_dataset, test_dataset


    @staticmethod
    def download_cifar10() -> (
        Tuple[
            torchvision.datasets.CIFAR10,
            torchvision.datasets.CIFAR10,
        ]
    ):
        """This function downloads the cifar10 dataset.

        Returns
        -------
            Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
            the train and test dataset
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

        cifar_train_ds = torchvision.datasets.CIFAR10(
            "../data/files",
            train=True,
            transform=preprocess,
            download=True,
        )

        cifar_test_ds = torchvision.datasets.CIFAR10(
            "../data/files",
            train=False,
            transform=preprocess,
            download=True,
        )
        return cifar_train_ds, cifar_test_ds

    ## Use this function to retrieve X, X, y arrays for training ML models
    @staticmethod
    def dataset_to_numpy(
        _df,
        _feature_cols: list,
        _metadata: dict,
        num_sensitive_features: int = 1,
        sensitive_features_last: bool = True,
    ):
        """Args:
        _df: pandas dataframe
        _feature_cols: list of feature column names
        _metadata: dictionary with metadata
        num_sensitive_features: number of sensitive features to use
        sensitive_features_last: if True, then sensitive features are encoded as last columns
        """

        # transform features to 1-hot
        _X = _df[_feature_cols]
        # take sensitive features separately
        print(
            f'Using {_metadata["protected_atts"][:num_sensitive_features]} as sensitive feature(s).'
        )
        if num_sensitive_features > len(_metadata["protected_atts"]):
            num_sensitive_features = len(_metadata["protected_atts"])
        _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]
        _X = _X.drop(columns=_metadata["protected_atts"][:num_sensitive_features])
        # 1-hot encode and scale features
        if "dummy_cols" in _metadata.keys():
            dummy_cols = _metadata["dummy_cols"]
        else:
            dummy_cols = None
        _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
        esc = MinMaxScaler()
        _X = esc.fit_transform(_X2)

        # current implementation assumes each sensitive feature is binary
        for i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
            assert len(_Z[tmp].unique()) == 2, "Sensitive feature is not binary!"

        # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
        _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
        # print(_Z2.head(), _Z2.shape)
        if sensitive_features_last:
            for i, tmp in enumerate(_Z.columns):
                assert (
                    _metadata["protected_att_values"][i] in _Z[tmp].unique()
                ), "Protected attribute value not found in data!"
                if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                    # swap columns
                    _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
        # change booleans to floats
        # _Z2 = _Z2.astype(float)
        # _Z = _Z2.to_numpy()
        _y = _df[_metadata["target_variable"]].values
        return _X, np.array([sv[0] for sv in _Z.values]), _y

    @staticmethod
    def download_dutch() -> (
        Tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
        ]
    ):
        """This function downloads the adult dataset.

        Returns
        -------
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            the train and test dataset
        """
        
        data = arff.loadarff("../data/dutch/dutch_census.arff")
        dutch_df = pd.DataFrame(data[0]).astype("int32")

        dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
        dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

        del dutch_df["sex"]
        del dutch_df["occupation"]

        dutch_df_feature_columns = [
            "age",
            "household_position",
            "household_size",
            "prev_residence_place",
            "citizenship",
            "country_birth",
            "edu_level",
            "economic_status",
            "cur_eco_activity",
            "Marital_status",
            "sex_binary",
        ]

        metadata_dutch = {
            "name": "Dutch census",
            "code": ["DU1"],
            "protected_atts": ["sex_binary"],
            "protected_att_values": [0],
            "protected_att_descriptions": ["Gender = Female"],
            "target_variable": "occupation_binary",
        }

        tmp = DatasetDownloader.dataset_to_numpy(dutch_df, dutch_df_feature_columns, metadata_dutch, num_sensitive_features=1)

        x = tmp[0]
        y = tmp[2]
        z = tmp[1]

        xyz = list(zip(x, y, z))
        random.shuffle(xyz)
        x, y, z = zip(*xyz)
        train_size = int(len(y) * 0.8)

        x_train = np.array(x[:train_size])
        x_test = np.array(x[train_size:])
        y_train = np.array(y[:train_size])
        y_test = np.array(y[train_size:])
        z_train = np.array(z[:train_size])
        z_test = np.array(z[train_size:])

        train_dataset = TabularDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(
                np.float32
            ),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )

        test_dataset = TabularDataset(
            x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
            z=z_test.astype(np.float32),
            y=y_test.astype(np.float32),
        )

        return train_dataset, test_dataset

    @staticmethod
    def download_adult() -> (
        Tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
        ]
    ):
        """This function downloads the adult dataset.

        Returns
        -------
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            the train and test dataset
        """
        path_train = "../data/adult/nuovi_dati/train"
        path_test = "../data/adult/nuovi_dati/test"
        train_data = datasets.DatasetFolder(path_train, np.load, ("npy"))
        test_data = datasets.DatasetFolder(path_test, np.load, ("npy"))

        samples = []
        targets = []
        for sample, target in train_data:
            samples.append(sample)
            targets.append(target)

        samples, targets = Utils.shuffle_lists(samples, targets)

        samples_test = []
        targets_test = []
        for sample, target in test_data:
            samples_test.append(sample)
            targets_test.append(target)

        samples_test, targets_test = Utils.shuffle_lists(samples_test, targets_test)

        adult_dataset_train = MyDataset(samples, targets)
        adult_dataset_test = MyDataset(samples_test, targets_test)

        return adult_dataset_train, adult_dataset_test
