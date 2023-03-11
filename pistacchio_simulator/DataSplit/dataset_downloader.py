import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from pistacchio_simulator.DataSplit.custom_dataset import CelebaGenderDataset, MyDataset
from pistacchio_simulator.Exceptions.errors import InvalidDatasetErrorNameError
from pistacchio_simulator.Utils.utils import Utils
import pandas as pd
from pathlib import Path


class DatasetDownloader:
    """This class is used to download some useful datasets.

    Raises
    ------
        InvalidDatasetErrorNameError: If the dataset name is not valid
    """

    @staticmethod
    def download_dataset(dataset_name: str, dataset_path: None | str = None) -> tuple:
        """Downloads the dataset with the given name.

        Args:
            dataset_name (str): The name of the dataset to download
        Raises:
            InvalidDatasetErrorNameError: If the dataset name is not valid

        Returns
        -------
            _type_: The downloaded dataset
        """
        match dataset_name:
            case "mnist":
                return DatasetDownloader.download_mnist()
            case "cifar10":
                DatasetDownloader.download_cifar10()
            case "adult":
                DatasetDownloader.download_adult()
            case "celeba":
                DatasetDownloader.download_celeba()
            case "celeba_gender":
                DatasetDownloader.download_celeba_gender()
            case "fashion_mnist":
                DatasetDownloader.download_fashion_mnist()
            case "imagenette":
                DatasetDownloader.download_imaginette()
            case "fair_face":
                DatasetDownloader.download_fair_face(dataset_path)
            case _:
                raise InvalidDatasetErrorNameError()

    @staticmethod
    def download_mnist() -> tuple[
        torchvision.datasets.MNIST,
        torchvision.datasets.MNIST,
    ]:
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
    def download_fashion_mnist() -> tuple[
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.FashionMNIST,
    ]:
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
    def download_imaginette() -> tuple[
        torchvision.datasets.ImageFolder,
        torchvision.datasets.ImageFolder,
    ]:
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

        imaginette_train_ds = torchvision.datasets.ImageFolder(
            "../data/imaginette/train",
            transform=transform,
        )

        imaginette_test_ds = torchvision.datasets.ImageFolder(
            "../data/imaginette/val",
            transform=transform,
        )
        return imaginette_train_ds, imaginette_test_ds

    @staticmethod
    def download_fair_face(
        dataset_path: str,
    ) -> tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder,]:
        """This function downloads the fair_face dataset.

        Returns
        -------
            Tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
            the train and test dataset
        """

        base_path = Path(dataset_path)
        train_df = pd.read_csv(f"{base_path}/train.csv")
        public_df = pd.read_csv(f"{base_path}/public.csv")
        test_df = pd.read_csv(f"{base_path}/test.csv")

        train_df["file"] = [f"{base_path}/{path}" for path in train_df["file"]]
        test_df["file"] = [f"{base_path}/{path}" for path in test_df["file"]]
        public_df["file"] = [f"{base_path}/{path}" for path in public_df["file"]]

        gender_map = {"Male": 0, "Female": 1}
        race_map = {
            "Black": 0,
            "East Asian": 1,
            "Indian": 2,
            "Latino_Hispanic": 3,
            "Middle Eastern": 4,
            "Southeast Asian": 5,
            "White": 6,
        }

        train_df["gender_code"] = [gender_map[x] for x in train_df["gender"]]
        test_df["gender_code"] = [gender_map[x] for x in test_df["gender"]]
        public_df["gender_code"] = [gender_map[x] for x in public_df["gender"]]

        train_df["race_code"] = [race_map[x] for x in train_df["race"]]
        test_df["race_code"] = [race_map[x] for x in test_df["race"]]
        public_df["race_code"] = [race_map[x] for x in public_df["race"]]
        return train_df, test_df, public_df

    @staticmethod
    def download_celeba() -> tuple[
        torchvision.datasets.CelebA,
        torchvision.datasets.CelebA,
    ]:
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((64, 64)),
            ],
        )
        train_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/train.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        test_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/test.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        return train_dataset, test_dataset

    @staticmethod
    def download_celeba_gender() -> tuple[
        torchvision.datasets.CelebA,
        torchvision.datasets.CelebA,
    ]:
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((64, 64)),
            ],
        )
        train_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/train_gender.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        test_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/test_gender.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        return train_dataset, test_dataset

    @staticmethod
    def download_cifar10() -> tuple[
        torchvision.datasets.CIFAR10,
        torchvision.datasets.CIFAR10,
    ]:
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

    @staticmethod
    def download_adult() -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
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
