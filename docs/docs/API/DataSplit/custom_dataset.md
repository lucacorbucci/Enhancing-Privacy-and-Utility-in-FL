---
sidebar_label: custom_dataset
title: custom_dataset
---

## MyDataset Objects

```python
class MyDataset(Dataset)
```

Definition of generic custom dataset.

#### \_\_init\_\_

```python
def __init__(samples: list, targets: list) -> None
```

Initialization of the dataset.

**Arguments**:

  ----
- `samples` _list_ - list of samples
- `targets` _list_ - list of targets

#### \_\_len\_\_

```python
def __len__() -> int
```

This function returns the size of the dataset.

Returns
-------
    _type_: size of the dataset

#### \_\_getitem\_\_

```python
def __getitem__(idx: int) -> Any
```

Returns a sample from the dataset.

**Arguments**:

- `idx` __type__ - index of the sample we want to retrieve
  
  Returns
  -------
- `_type_` - sample we want to retrieve

## CelebaGenderDataset Objects

```python
class CelebaGenderDataset(Dataset)
```

Definition of the dataset used for the Celeba Dataset.

#### \_\_init\_\_

```python
def __init__(csv_path: str,
             image_path: str,
             transform: torchvision.transforms = None) -> None
```

Initialization of the dataset.

**Arguments**:

  ----
- `csv_path` _str_ - path of the csv file with all the information
  about the dataset
- `image_path` _str_ - path of the images
- `transform` _torchvision.transforms, optional_ - Transformation to apply
  to the images. Defaults to None.

#### \_\_getitem\_\_

```python
def __getitem__(index: int) -> tuple[torch.Tensor, torch.Tensor]
```

Returns a sample from the dataset.

**Arguments**:

- `idx` __type__ - index of the sample we want to retrieve
  
  Returns
  -------
- `_type_` - sample we want to retrieve

#### \_\_len\_\_

```python
def __len__() -> int
```

This function returns the size of the dataset.

Returns
-------
    int: size of the dataset

