---
sidebar_label: generate_dataset
title: generate_dataset
---

This file is used to generate the dataset for the experiments.

#### print\_debug

```python
def print_debug(counters: Counter) -> None
```

This prints the stats of the dataset.

**Arguments**:

  ----
- `counters` __type__ - The stats of the dataset

#### convert\_targets\_to\_int

```python
def convert_targets_to_int(percentage_configuration: dict,
                           targets: list) -> None
```

Converts the targets that we specified in the percentage_configuration
from str to int when the targets of the dataset are integers.

**Arguments**:

  ----
- `percentage_configuration` _dict_ - the percentage configuration specified by the user
- `targets` _list_ - target list of the dataset we want to split

#### get\_dataset

```python
def get_dataset(config: Preferences, custom_dataset: dict = None)
```

This function is used to download the dataset based on the
preferences that we pass as parameter.

**Arguments**:

- `config` _Preferences_ - preferences of the experiment
  
  Returns
  -------
  Tuple[torch.Dataset, torch.Dataset]: _description_

#### pre\_process\_dataset

```python
def pre_process_dataset(train_ds, percentage_underrepresented_classes,
                        underrepresented_classes, num_reduced_nodes,
                        num_samples_underrepresented_classes) -> None
```

_summary_.

**Arguments**:

  ----
- `train_ds` __type__ - _description_
- `test_ds` __type__ - _description_
- `percentage_underrepresented_classes` __type__ - _description_
- `underrepresented_classes` __type__ - _description_
- `num_reduced_nodes` __type__ - _description_
- `num_samples_underrepresented_classes` __type__ - _description_

#### stratified

```python
def stratified(train_ds, test_ds, num_clusters, num_nodes,
               max_samples_per_cluster)
```

_summary_.

**Arguments**:

- `train_ds` __type__ - _description_
- `test_ds` __type__ - _description_
- `num_clusters` __type__ - _description_
- `num_nodes` __type__ - _description_
- `max_samples_per_cluster` __type__ - _description_
  
  Returns
  -------
- `_type_` - _description_

#### stratified\_with\_some\_reduced

```python
def stratified_with_some_reduced(train_ds, test_ds, num_clusters, num_nodes,
                                 num_reduced_nodes, max_samples_per_cluster,
                                 underrepresented_classes,
                                 percentage_underrepresented_classes)
```

_summary_.

**Arguments**:

- `train_ds` __type__ - _description_
- `test_ds` __type__ - _description_
- `num_clusters` __type__ - _description_
- `num_nodes` __type__ - _description_
- `num_reduced_nodes` __type__ - _description_
- `max_samples_per_cluster` __type__ - _description_
- `underrepresented_classes` __type__ - _description_
- `percentage_underrepresented_classes` __type__ - _description_
  
  Returns
  -------
- `_type_` - _description_

#### percentage\_max\_samples

```python
def percentage_max_samples(config, train_ds, test_ds, num_clusters,
                           max_samples_per_cluster)
```

_summary_.

**Arguments**:

- `config` __type__ - _description_
- `train_ds` __type__ - _description_
- `test_ds` __type__ - _description_
- `num_clusters` __type__ - _description_
- `max_samples_per_cluster` __type__ - _description_
  
  Returns
  -------
- `_type_` - _description_

#### percentage

```python
def percentage(config, train_ds, test_ds, num_nodes, task, names)
```

_summary_.

**Arguments**:

  ----
- `config` __type__ - _description_
- `train_ds` __type__ - _description_
- `test_ds` __type__ - _description_
- `num_nodes` __type__ - _description_
- `task` __type__ - _description_
- `names` __type__ - _description_

#### store\_on\_disk

```python
def store_on_disk(config, cluster_datasets, cluster_datasets_test, test_ds,
                  names)
```

_summary_.

**Arguments**:

  ----
- `config` __type__ - _description_
- `cluster_datasets` __type__ - _description_
- `cluster_datasets_test` __type__ - _description_
- `test_ds` __type__ - _description_
- `names` __type__ - _description_

#### generate\_splitted\_dataset

```python
def generate_splitted_dataset(config: Preferences,
                              custom_dataset: dict = None) -> None
```

This function is used to generate the dataset based on the
configuration file passed as parameter.

**Arguments**:

- `config` __type__ - configuration file
  
  Raises
  ------
- `InvalidSplitType` - If we select a split type that is not valid

#### main

```python
def main() -> None
```

Based on the preferences, this function generates the dataset.

Raises
------
    InvalidSplitType: If the split type is not valid

