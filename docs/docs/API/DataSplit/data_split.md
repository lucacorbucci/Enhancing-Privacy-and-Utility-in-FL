---
sidebar_label: data_split
title: data_split
---

## DataSplit Objects

```python
class DataSplit()
```

This class is used to split the dataset in different ways.

#### convert\_subset\_to\_dataset

```python
@staticmethod
def convert_subset_to_dataset(subset: Subset) -> tuple[list, list]
```

This function converts a subset of data to a dataset.

**Arguments**:

  ----
- `subset` __type__ - the subset of data to be converted to a dataset

**Returns**:

  Tuple[List, List]: the converted dataset

#### create\_splits

```python
@staticmethod
def create_splits(
    dataset: torch.utils.data.Dataset,
    num_workers: int,
    max_samples_per_cluster: list | int | None = None
) -> tuple[list[int], list[int]]
```

This function returns a list that will be used to split the dataset.
If the parameter max_samples_per_cluster is equal to -1 then it considers
the size of the dataset and then splits it in equal parts.
Otherwise, if we pass a value different than -1, it will create
a splitting configuration where each cluster will receive
max_samples_per_cluster samples.

Raises
------
ValueError: it raises this error when we choose
a value for max_samples_per_cluster that is too big.
In particular, if max_samples_per_cluster * num_workers &gt; len(dataset)
then the function will raise an exception.
It can also raise ValueError when the dataset passed as
parameter is empty.

**Arguments**:

- `dataset` _torch.utils.data.Dataset_ - the dataset we want to split
- `num_workers` _int_ - number of nodes
- `max_samples_per_cluster` _int_ - default value == -1. The maximum
  amount of samples per cluster
  

**Example**:

  &gt;&gt;&gt; dataset = MyDataset(
  samples=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100,
  targets=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 100,
  )
  &gt;&gt;&gt;
  
  Returns
  -------
  Tuple[List[int], List[int]]: the first list is a list of sizes of the splits
  the second list can be empty if max_samples_per_cluster == -1 or it contains
  the remaining data that could not be distributed equally among the clusters.

#### split\_dataset

```python
@staticmethod
def split_dataset(dataset: torch.utils.data.Dataset,
                  num_workers: int,
                  max_samples_per_cluster: list | int | None = None) -> Any
```

This function splits the dataset in equal parts. If max_samples_per_cluster
is different than -1 then it will split the dataset in a way that each cluster
will receive max_samples_per_cluster samples.

**Arguments**:

- `dataset` _torch.utils.data.Dataset_ - the dataset we want to split
- `num_workers` _int_ - number of nodes of the dataset
  
  Returns
  -------
- `List[torch.utils.data.DataLoader]` - the splitted dataset

#### merge\_indices

```python
@staticmethod
def merge_indices(datasets: list) -> torch.utils.data.DataLoader
```

This function merges the indices of a list of datasets
into a single list.

**Arguments**:

- `datasets` _list_ - the list of datasets to merge
  
  Returns
  -------
- `list` - the merged list of indices

#### check\_percentage\_validity

```python
@staticmethod
def check_percentage_validity(data: torch.utils.data.DataLoader,
                              percentage_configuration: dict) -> None
```

This function checks if the percentage configuration is valid.

**Arguments**:

- `data` _torch.utils.data.DataLoader_ - data we want to split
- `percentage_configuration` _dict_ - configuration of the percentage
  
  Raises
  ------
- `InvalidSplitConfiguration` - It is raised when the percentage is not valid

#### generate\_classes\_dictionary

```python
@staticmethod
def generate_classes_dictionary(data: torch.utils.data.DataLoader) -> dict
```

Generates a dictionary with all the possible classes as key and for
each key all the possible values
{&quot;class_0&quot;: [data]}.

**Arguments**:

- `data` _torch.utils.data.DataLoader_ - data we want to split
  
  Returns
  -------
- `dict` - _description_

#### generate\_aggregated\_percentages

```python
@staticmethod
def generate_aggregated_percentages(data: list) -> dict
```

Generates a dictionary with all the possible classes and
for each class the corresponding aggregated percentages
{&quot;class_0&quot;: {&quot;user_0&quot;: 40, &quot;user_1&quot;: 20, &quot;user_2&quot;:40}}.

**Arguments**:

- `data` _list_ - percentage distribution
  
  Returns
  -------
- `dict` - the new aggregated dictionary

#### sample\_list\_by\_percentage

```python
@staticmethod
def sample_list_by_percentage(class_percentage: dict, indices: list) -> dict
```

This function split the list indices passed as parameter
in N parts. The size of each part depends on the percentage
that we specify in class_percentage and that correspond to
each user.

**Arguments**:

  ----
- `class_percentage` __type__ - percentage of data of each user
- `indices` __type__ - list of indices we want to split among users

#### generate\_aggregated\_indices

```python
@staticmethod
def generate_aggregated_indices(percentages: dict, indices: dict) -> dict
```

# Inside samples we have the classes as keys and the
# corresponding dict of users and indices as values
# Example: {0: {&quot;user_1&quot;: [...], &quot;user_2&quot;: [...], &quot;user_3&quot;: [...]}}.

**Arguments**:

- `percentages` _dict_ - _description_
- `indices` _dict_ - _description_
  
  Returns
  -------
- `dict` - _description_

#### aggregate\_indices\_by\_cluster

```python
@staticmethod
def aggregate_indices_by_cluster(samples: dict) -> dict
```

This function aggregates the indices by cluster.
Originally, they are aggregated by class, we want to merge the
N lists of indices for each class in a single list of indices.

Example: {&quot;cluster_0&quot;: [1,2,3,4,5,6.....]}

**Arguments**:

- `samples` _dict_ - a dictionary with all the indices of each cluster
  divided by class
  
  Returns
  -------
- `dict` - a dictionary with all the indices of each cluster

#### split\_cluster\_dataset\_in\_parts

```python
@staticmethod
def split_cluster_dataset_in_parts(
        data: torch.utils.data.DataLoader, cluster_names: list,
        splitted_indices: dict) -> tuple[list, list]
```

This function takes as input the dataset that we want to split,
a list with the names of the clusters and a dictionary where we store
as key the cluster names and as values a list of N lists. Each
of these lists contains the indices of the dataset that we
want to assign to the each user.

**Arguments**:

- `data` _torch.utils.data.DataLoader_ - the dataset we want to split
- `cluster_names` _list_ - the names of the clusters
- `splitted_indices` _dict_ - the dictionary with the indices of each cluster
  

**Example**:

  &gt;&gt;&gt; cluster_names = [&quot;cluster1&quot;, &quot;cluster2&quot;]
  &gt;&gt;&gt; splitted_indices = {
- `&quot;cluster1&quot;` - [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]],
- `&quot;cluster2&quot;` - [[11, 12], [13, 14]],
  }
  &gt;&gt;&gt; mock_dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
  &gt;&gt;&gt; cluster_datasets, counter = DataSplit.split_cluster_dataset_in_parts(
  mock_dataloader, cluster_names, splitted_indices
  )
  
  Returns
  -------
  Tuple[list, list]: the first element of the tuple is a list of
  torch.utils.data.Subset objects, each of them is the subset
  that is assigned to a user. The second element is a list of
  Counters, each of them contains the number of elements of each
  class that are assigned to a user.
  first element: [&lt;torch.utils.data.dataset.Subset object at 0x155314c70&gt;,
  &lt;torch.utils.data.dataset.Subset object at 0x155314cd0&gt;,
  &lt;torch.utils.data.dataset.Subset object at 0x155314d30&gt;,
  &lt;torch.utils.data.dataset.Subset object at 0x155314d90&gt;,
  &lt;torch.utils.data.dataset.Subset object at 0x155314df0&gt;]
  second element: [Counter({0: 4}), Counter({1: 4}),
- `Counter({2` - 3}), Counter({0: 2}), Counter({1: 2})]

#### aggregate\_indices\_by\_class

```python
@staticmethod
def aggregate_indices_by_class(targets_cluster: list, indices: list) -> dict
```

Aggregates the indices assigned to the cluster by class.
Returns a dictionary with the indices of each class.

**Arguments**:

- `targets_cluster` _list_ - _description_
- `indices` _list_ - _description_
  

**Example**:

  &gt;&gt;&gt; splitted_indices = [1, 2, 3, 4, 5, 6]
  &gt;&gt;&gt; targets = [0, 1, 2, 0, 0, 1]
  &gt;&gt;&gt; aggregated_indices = DataSplit.aggregate_indices_by_class(targets,
  splitted_indices)
- `{0` - [1, 4, 5], 1: [2, 6], 2: [3]}
  
  Returns
  -------
- `dict` - A dictionary with the iaggregated_percentagesndices of each class
- `{&quot;class_0&quot;` - [1,2,3,4,5,6,7,8,9,10],
- `&quot;class_1&quot;` - [11,12,13,14,15,16,17,18,19,20]}

#### create\_percentage\_subsets

```python
@staticmethod
def create_percentage_subsets(
        splitted_indices: dict, nodes_names: list, targets: np.ndarray,
        data: torch.utils.data.DataLoader) -> tuple[list, list]
```

This function creates the subsets of the original dataset
for a cluster. Given the dataset, it is splitted in N parts
where N is the number of users in the cluster.

**Example**:

  &gt;&gt;&gt; splitted_indices = {
- `&quot;cluster_0_user_0&quot;` - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
- `&quot;cluster_0_user_1&quot;` - [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
- `&quot;cluster_0_user_2&quot;` - [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
- `&quot;cluster_0_user_3&quot;` - [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
- `&quot;cluster_0_user_4&quot;` - [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
  }
  &gt;&gt;&gt; nodes_names = [
  &quot;cluster_0_user_0&quot;,
  &quot;cluster_0_user_1&quot;,
  &quot;cluster_0_user_2&quot;,
  &quot;cluster_0_user_3&quot;,
  &quot;cluster_0_user_4&quot;,
  ]
  &gt;&gt;&gt; data = DataLoader(dataset, batch_size=1, shuffle=True)
  &gt;&gt;&gt; cluster_datasets, counter = DataSplit.create_percentage_subsets(
  data=data,
  splitted_indices=splitted_indices,
  nodes_names=nodes_names,
  targets=np.array(targets),
  )
  ([&lt;torch.utils.data.dataset.Subset object at 0x140689640&gt;,
  &lt;torch.utils.data.dataset.Subset object at 0x140689550&gt;, ...],
- `[Counter({4` - 2, 0: 2, 8: 2, 5: 1, 3: 1, 6: 1, 1: 1}),
- `Counter({1` - 3, 4: 2, 7: 2, 0: 2, 3: 1}), ..... )])
  

**Arguments**:

- `splitted_indices` _dict_ - is a dictionary
  with the names of the nodes as keys and the lists of the indices
  that will be assigned to each node as values.
- `nodes_names` _list_ - contains the names of the users in the cluster.
- `targets` _np.ndarray_ - contains the targets of the dataset.
- `data` _torch.utils.data.DataLoader_ - the original dataset we want to split
  
  Returns
  -------
  Tuple[list, list]: a list that contains the subsets created
  from the original dataset and assigned to each user in the cluster and
  a list that contains the counters of the targets of each subset.

#### split\_cluster\_dataset\_by\_percentage

```python
@staticmethod
def split_cluster_dataset_by_percentage(
        nodes_distribution: dict, data: torch.utils.data.DataLoader,
        splitted_indices: dict) -> tuple[list, list]
```

This function is used to split the dataset in N parts
where N is the number of clusters. Then each part
is splitted in M parts where M is the number of users.

**Example**:

  &gt;&gt;&gt; nodes_distribution = {
- `&quot;cluster_0&quot;` - {
- `&quot;user_0&quot;` - {0: 40, 1: 20},
- `&quot;user_1&quot;` - {0: 40, 1: 40},
- `&quot;user_2&quot;` - {0: 20, 1: 40},
  },
- `&quot;cluster_1&quot;` - {
- `&quot;user_0_1&quot;` - {0: 40, 1: 20},
- `&quot;user_1_1&quot;` - {0: 40, 1: 40},
- `&quot;user_2_1&quot;` - {0: 20, 1: 40},
  },
  }
  &gt;&gt;&gt; splitted_indices = {
- `&quot;cluster_0&quot;` - [0, 1, 2, 3, 4],
- `&quot;cluster_1&quot;` - [5, 6, 7, 8, 9],
  }
  &gt;&gt;&gt; data_loader = DataLoader(mock_dataset, batch_size=1)
  &gt;&gt;&gt; cluster_datasets, counters = DataSplit.split_cluster_dataset_by_percentage(
  nodes_distribution, data_loader, splitted_indices
  )
  ([&lt;torch.utils.data.dataset.Subset object at 0x14491a100&gt;,
  &lt;torch.utils.data.dataset.Subset object at 0x14491a130&gt;, .....],
  [Counter(), Counter({0: 1, 1: 1}), Counter({1: 2, 0: 1}),....)])
  

**Arguments**:

- `nodes_distribution` _dict_ - the configuration that we want to
  use in each cluster and for each user.
- `data` _torch.utils.data.DataLoader_ - the dataset we want to split.
- `splitted_indices` _dict_ - a dictionary with the names of the clusters
  as keys and the lists of the indices that will be assigned to each
  cluster as values.
  
  Returns
  -------
  Tuple[list, list]: a list that contains the subsets created
  from the original dataset and assigned to each user in the cluster and
  a list that contains the counters of the targets of each subset.

#### random\_class\_distribution

```python
@staticmethod
def random_class_distribution(num_nodes: int, total_sum: int) -> list
```

This function generates a random distribution of percentages
for the classes of the cluster. It considers the number of the nodes in
the cluster and splits the data in N parts where N is the number of nodes.
Example: in cluster 0 we have user 0 with the following distribution:
[40, 0, 20, 20, 0, 0, 0, 20]
When the percentage value is 0, we will not consider that class for that user.


**Arguments**:

- `num_nodes` _int_ - number of nodes of the cluster
- `total_sum` _int_ - total sum of the percentages of the classes
  
  Returns
  -------
- `_type_` - _description_

#### generate\_nodes\_distribution

```python
@staticmethod
def generate_nodes_distribution(num_nodes: int, classes: list,
                                names: list) -> dict
```

This function generates the final distribution of the classes of the cluster.
In particular, we consider the number of nodes in the cluster and the number of
classes assigned to the cluster.
For each of the classes, we assign a certain percentage of data for each node.
Every time we generate a distribution, we check if we have a node without any
class assignment. If this happens, we generate a new distribution.

**Arguments**:

- `num_nodes` _int_ - number of nodes in the cluster
- `classes` _list_ - classes assigned to the cluster
- `names` _list_ - names of the nodes inside the cluster
  
  Returns
  -------
- `_type_` - _description_

#### remove\_indices\_from\_dataset

```python
@staticmethod
def remove_indices_from_dataset(dataset: torch.utils.data.Dataset,
                                indices_to_remove: np.ndarray) -> None
```

This function removes the indices passed as
parameter from a dataset.

**Arguments**:

  ----
- `dataset` _torch.utils.data.Dataset_ - the dataset we want to modify
- `indices_to_remove` _list_ - the indices we want to remove from the dataset

#### reduce\_samples

```python
@staticmethod
def reduce_samples(
        dataset: torch.utils.data.Dataset,
        classes: list,
        percentage_underrepresented_classes: list | None = None,
        num_samples_underrepresented_classes: list | None = None) -> None
```

This function reduces the samples of the classes indicated
in the parameters.

**Arguments**:

  ----
- `dataset` _torch.utils.data.Dataset_ - the dataset we want to reduce
- `classes` _list_ - the classes we want to reduce
- `percentage_underrepresented_classes` _list, optional_ - the percentage of
  samples of the specified class that we want to remove
  from the dataset . Defaults to []. num_samples_underrepresented_classes
  (list, optional): The amount of samples that
  we want for each of the classes we want to reduce. Defaults to [].

#### reduce\_samples\_by\_percentage

```python
@staticmethod
def reduce_samples_by_percentage(
        dataset: torch.utils.data.Dataset, classes: list,
        percentage_underrepresented_classes: list) -> None
```

This function reduces the samples of the classes indicated
in the list classes by the percentages indicated in the list
percentage_underrepresented_classes.

**Arguments**:

  ----
- `dataset` _torch.utils.data.Dataset_ - the dataset we want to reduce
- `classes` _list_ - the classes we want to reduce
- `percentage_underrepresented_classes` _list_ - the percentage of samples
  of the specified class that we want to remove from the dataset

#### reduce\_samples\_by\_num\_samples

```python
@staticmethod
def reduce_samples_by_num_samples(
        dataset: torch.utils.data.Dataset, classes: list,
        num_samples_underrepresented_classes: list) -> None
```

This function reduces the samples of the classes indicated
in the list classes to the number of samples indicated in the list
num_samples_underrepresented_classes.

**Arguments**:

  ----
- `dataset` _torch.utils.data.Dataset_ - the dataset we want to reduce
- `classes` _list_ - the classes we want to reduce
- `num_samples_underrepresented_classes` _list_ - the amount of samples
  of the specified class that we want to keep in the dataset

#### stratified\_sampling

```python
@staticmethod
def stratified_sampling(
    dataset: torch.utils.data.DataLoader,
    num_workers: int,
    max_samples_per_cluster: list | int | None = None
) -> tuple[Any, list[Counter]]
```

This function performs stratified sampling on the dataset.

**Arguments**:

- `dataset` _torch.utils.data.DataLoader_ - the dataset to sample from
- `num_workers` _int_ - number of users that will be created
  
  
  Returns
  -------
  Tuple[Any, Counter]: The splitted dataset and a counter with the
  number of samples per cluster

#### stratified\_sampling\_with\_some\_nodes\_reduced

```python
@staticmethod
def stratified_sampling_with_some_nodes_reduced(
    dataset: torch.utils.data.DataLoader, num_workers: int,
    num_reduced_nodes: int, max_samples_per_cluster: int,
    underrepresented_classes: list[int],
    percentage_underrepresented_classes: list[float]
) -> tuple[Any, list[Counter]]
```

This function performs stratified sampling on the dataset.

**Arguments**:

- `dataset` _torch.utils.data.DataLoader_ - the dataset to sample from
- `num_workers` _int_ - number of users that will be created
- `num_reduced_nodes` _int_ - number of users that will have a reduced dataset
- `max_samples_per_cluster` _int_ - maximum number of samples per node.
  
  Returns
  -------
  Tuple[Any, Counter]: The splitted dataset and a counter with the
  number of samples per cluster

#### percentage\_split

```python
@staticmethod
def percentage_split(
        dataset: torch.utils.data.DataLoader,
        percentage_configuration: dict,
        task: Task,
        num_workers: int = 0,
        nodes_distribution: dict | None = None) -> tuple[list, list]
```

This function is used to split the original dataset based on a
configuration passed as a parameter.
We can split the dataset in two ways:
- If we don&#x27;t provide the paramert nodes_distribution, we
will split the dataset in N parts (N is the number of cluster)
and then for each cluster we will split again the dataset in
M parts (M is the number of users in the cluster). In this case
the distribution will be non iid for the clusters
and iid for the users.
- If we provide a nodes_distribution, we will split the dataset
in N parts (N is the number of cluster) and then for each cluster
we will split again the dataset in M parts (M is the number of users
in the cluster). In this case we will use a percentage split
both for the clusters and for the users. If we provide a proper
nodes_distribution dictionary, we will have both a non iid
distribution for the clusters and a non iid distribution for the users.

**Example**:

  &gt;&gt;&gt; percentage_configuration = {
- `&quot;cluster_0&quot;` - {0: 60, 1: 30, 2: 20, 3: 20},
- `&quot;cluster_1&quot;` - {1: 70, 2: 40, 3: 20, 4: 20},
- `&quot;cluster_2&quot;` - {2: 40, 3: 20, 4: 20, 5: 20},
- `&quot;cluster_3&quot;` - {3: 40, 4: 20, 5: 20, 6: 30},
- `&quot;cluster_4&quot;` - {4: 40, 5: 20, 6: 30, 7: 10},
- `&quot;cluster_5&quot;` - {5: 40, 6: 20, 7: 30, 8: 30},
- `&quot;cluster_6&quot;` - {6: 20, 7: 40, 8: 30, 9: 70},
- `&quot;cluster_7&quot;` - {7: 20, 8: 40, 9: 30, 0: 40},
  }
  &gt;&gt;&gt; nodes_distribution = {
- `&quot;cluster_0&quot;` - {
- `&quot;cluster_0_user_0&quot;` - {0: 18, 1: 15, 2: 19, 3: 36},
- `&quot;cluster_0_user_1&quot;` - {0: 9, 1: 40, 2: 5, 3: 8},
- `&quot;cluster_0_user_2&quot;` - {0: 30, 1: 18, 2: 20, 3: 38},
- `&quot;cluster_0_user_3&quot;` - {0: 28, 1: 17, 2: 27, 3: 10},
- `&quot;cluster_0_user_4&quot;` - {0: 15, 1: 10, 2: 29, 3: 8},
  },
- `&quot;cluster_1&quot;` - {
- `&quot;cluster_1_user_0&quot;` - {1: 23, 2: 8, 3: 28, 4: 23},
- `&quot;cluster_1_user_1&quot;` - {1: 30, 2: 10, 3: 12, 4: 23},
- `&quot;cluster_1_user_2&quot;` - {1: 25, 2: 21, 3: 29, 4: 15},
- `&quot;cluster_1_user_3&quot;` - {1: 2, 2: 25, 3: 10, 4: 28},
- `&quot;cluster_1_user_4&quot;` - {1: 20, 2: 36, 3: 21, 4: 11},
  }, .....
  }
  &gt;&gt;&gt; samples = list(np.random.rand(10000, 10))
  &gt;&gt;&gt; targets = list(np.random.randint(0, 10, size=10000))
  &gt;&gt;&gt; mock_dataset = MyDataset(samples, targets)
  &gt;&gt;&gt; cluster_datasets, counter = DataSplit.percentage_split(
  data=data,
  percentage_configuration=percentage_configuration,
  num_workers=2,
  nodes_distribution=nodes_distribution,
  task=&quot;federated&quot;
  )
  ([&lt;torch.utils.data.dataset.Subset object at 0x14b0434c0&gt;,
  &lt;torch.utils.data.dataset.Subset object at 0x14b043520&gt;, .....],
- `[Counter({0` - 35, 3: 24, 1: 15, 2: 14}),
- `Counter({1` - 40, 0: 18, 3: 5, 2: 4}), ......])
  

**Arguments**:

- `data` _torch.utils.data.DataLoader_ - the dataset we want to split
- `percentage_configuration` _dict_ - the configuration of the cluster we
  want to create
- `num_workers` _int, optional_ - _description_. Defaults to 0. The number of
  users per cluster
- `nodes_distribution` _dict, optional_ - _description_. Defaults to None. The
  distribution of the users per cluster
- `task` _Task_ - the task we want to perform after the split. It is important
  to specify it because for some tasks we don&#x27;t want to check the percentage
  validity of the configuration passed as parameter.
  
  Returns
  -------
  Tuple[list, list]: the first returned element is the list with
  the subsets that were created during the process. The second
  returned element is a list with the counters for each cluster,
  this is useful to understand how many samples are assigned to each user.

#### percentage\_sampling\_max\_samples

```python
@staticmethod
def percentage_sampling_max_samples(dataset: torch.utils.data.DataLoader,
                                    percentage_configuration: dict,
                                    num_workers: int,
                                    max_samples_per_cluster: list | int,
                                    num_nodes: int = 1) -> tuple[list, list]
```

This method is used to split the dataset using the percentage sampling.
For each cluster we specify the percentage of samples we want in the cluster.
For instance if we have 10 classes we could specify that we want 10% of each
class in cluster 0. The percentage is specified as a dictionary where the key
is the name of the cluster and the value is a dictionary with the percentage
of each class. For instance:
&quot;percentage_configuration&quot;: {
&quot;cluster_0&quot;: {
&quot;0&quot;: 10,
&quot;1&quot;: 10,
&quot;2&quot;: 10,
&quot;3&quot;: 10,
&quot;4&quot;: 10,
&quot;5&quot;: 10,
&quot;6&quot;: 10,
&quot;7&quot;: 10,
&quot;8&quot;: 10,
&quot;9&quot;: 10
},
}.

**Example**:

  
  

**Arguments**:

- `dataset` _torch.utils.data.DataLoader_ - the dataset we want to split
- `percentage_configuration` _dict_ - the configuration of the cluster
  we want to create
- `num_workers` _int_ - total number of clusters we want to have
- `max_samples_per_cluster` _Union[list, int]_ - maximum number of samples
  we want inside each cluster this can be a list when we want to specify
  the size of each cluster or a int when we specify
  a size that is the same for all the clusters
  
  Raises
  ------
- `ValueError` - When the length of max_samples_per_cluster is
  not equal to num_workers
- `ValueError` - When the sum of max_samples_per_cluster is greater
  than the length of the dataset

#### percentage\_sampling

```python
@staticmethod
def percentage_sampling(dataset: torch.utils.data.DataLoader,
                        percentage_configuration: dict,
                        num_nodes: int = 1) -> tuple[list, list]
```

This method is used to split the dataset using the percentage sampling.
For each cluster we specify the percentage of samples we want in the cluster.
For instance if we have 10 classes we could specify that we want 10% of each
class in cluster 0. The percentage is specified as a dictionary where the key
is the name of the cluster and the value is a dictionary with the percentage
of each class. For instance:
&quot;percentage_configuration&quot;: {
&quot;cluster_0&quot;: {
&quot;0&quot;: 10,
&quot;1&quot;: 10,
&quot;2&quot;: 10,
&quot;3&quot;: 10,
&quot;4&quot;: 10,
&quot;5&quot;: 10,
&quot;6&quot;: 10,
&quot;7&quot;: 10,
&quot;8&quot;: 10,
&quot;9&quot;: 10
},
}.

**Example**:

  
  

**Arguments**:

- `dataset` _torch.utils.data.DataLoader_ - the dataset we want to split
- `percentage_configuration` _dict_ - the configuration of the
  cluster we want to create
- `num_workers` _int_ - total number of clusters we want to have
  
  Raises
  ------
- `ValueError` - When the configuration of the cluster is not correct

