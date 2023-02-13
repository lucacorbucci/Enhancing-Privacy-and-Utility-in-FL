---
sidebar_label: storage_manager
title: storage_manager
---

## StorageManager Objects

```python
class StorageManager()
```

This class is used to store the model on disk.

#### write\_splitted\_dataset

```python
@staticmethod
def write_splitted_dataset(dataset_name: str, splitted_dataset: list,
                           dataset_type: str, names: list) -> None
```

This function writes the splitted dataset in a pickle file
and then stores it on disk.

**Arguments**:

  ----
- `dataset_name` _str_ - name of the dataset
- `splitted_dataset` _Any_ - list of splitted dataset
- `dataset_type` _str_ - Type of the dataset. i.e train, test, validation
- `names` _List[str]_ - names of the nodes

#### write\_validation\_dataset

```python
@staticmethod
def write_validation_dataset(dataset_name: str, dataset: Any,
                             dataset_type: str) -> None
```

This is used to write the validation dataset on disk.

**Arguments**:

  ----
- `dataset_name` _str_ - The name of the dataset
- `dataset` _Any_ - The dataset to write
- `dataset_type` _str_ - The type of the dataset

