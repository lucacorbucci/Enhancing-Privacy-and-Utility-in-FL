---
sidebar_label: dataset_downloader
title: dataset_downloader
---

## DatasetDownloader Objects

```python
class DatasetDownloader()
```

This class is used to download some useful datasets.

Raises
------
    InvalidDatasetName: If the dataset name is not valid

#### download\_dataset

```python
@staticmethod
def download_dataset(dataset_name: str) -> tuple
```

Downloads the dataset with the given name.

**Arguments**:

- `dataset_name` _str_ - The name of the dataset to download

**Raises**:

- `InvalidDatasetName` - If the dataset name is not valid
  
  Returns
  -------
- `_type_` - The downloaded dataset

#### download\_mnist

```python
@staticmethod
def download_mnist(
) -> tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST, ]
```

This function downloads the mnist dataset.

Returns
-------
    Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    the train and test dataset

#### download\_fashion\_mnist

```python
@staticmethod
def download_fashion_mnist() -> tuple[torchvision.datasets.FashionMNIST,
                                      torchvision.datasets.FashionMNIST, ]
```

This function downloads the mnist dataset.

Returns
-------
    Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    the train and test dataset

#### download\_imaginette

```python
@staticmethod
def download_imaginette() -> tuple[torchvision.datasets.ImageFolder,
                                   torchvision.datasets.ImageFolder, ]
```

This function downloads the mnist dataset.

Returns
-------
    Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    the train and test dataset

#### download\_celeba

```python
@staticmethod
def download_celeba(
) -> tuple[torchvision.datasets.CelebA, torchvision.datasets.CelebA, ]
```

This function downloads the mnist dataset.

Returns
-------
    Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    the train and test dataset

#### download\_celeba\_gender

```python
@staticmethod
def download_celeba_gender(
) -> tuple[torchvision.datasets.CelebA, torchvision.datasets.CelebA, ]
```

This function downloads the mnist dataset.

Returns
-------
    Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    the train and test dataset

#### download\_cifar10

```python
@staticmethod
def download_cifar10(
) -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10, ]
```

This function downloads the cifar10 dataset.

Returns
-------
    Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    the train and test dataset

#### download\_adult

```python
@staticmethod
def download_adult(
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, ]
```

This function downloads the adult dataset.

Returns
-------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    the train and test dataset

