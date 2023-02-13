---
sidebar_label: federated_model
title: federated_model
---

## FederatedModel Objects

```python
class FederatedModel(ABC, Generic[TDestination])
```

This class is used to create the federated model that
we will train. It returns a different model based on the
dataset we want to use.

#### \_\_init\_\_

```python
def __init__(dataset_name: str,
             node_name: str,
             preferences: Preferences | None = None) -> None
```

Initialize the Federated Model.

**Arguments**:

- `dataset_name` _str_ - Name of the dataset we want to use
- `node_name` _str_ - name of the node we are working on
- `preferences` _Preferences, optional_ - Configuration for this run. Defaults to None.
  
  Raises
  ------
- `InvalidDatasetName` - _description_

#### init\_model

```python
def init_model(net: nn.Module) -> None
```

Initialize the Federated Model before the use of it.

**Arguments**:

  ----
- `net` _nn.Module_ - model we want to use

#### add\_model

```python
def add_model(model: nn.Module) -> None
```

This function adds the model passed as parameter
as the model used in the FederatedModel.

**Arguments**:

  ----
- `model` _nn.Module_ - Model we want to inject in the Federated Model

#### load\_data

```python
def load_data(
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
```

Load training and test dataset.

Returns
-------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: training and test set

Raises
------
    Exception: Preference is not initialized

#### print\_data\_stats

```python
def print_data_stats(trainloader: torch.utils.data.DataLoader) -> None
```

Debug function used to print stats about the loaded datasets.

**Arguments**:

  ----
- `trainloader` __type__ - training set

#### get\_weights\_list

```python
def get_weights_list() -> list[float]
```

Get the parameters of the network.

Raises
------
    Exception: if the model is not initialized it raises an exception

Returns
-------
    List[float]: parameters of the network

#### get\_weights

```python
def get_weights() -> TDestination
```

Get the weights of the network.

Raises
------
    Exception: if the model is not initialized it raises an exception

Returns
-------
    _type_: weights of the network

#### update\_weights

```python
def update_weights(avg_tensors: TDestination) -> None
```

This function updates the weights of the network.

Raises
------
Exception: _description_

**Arguments**:

- `avg_tensors` __type__ - tensors that we want to use in the network

#### store\_model\_on\_disk

```python
def store_model_on_disk() -> None
```

This function is used to store the trained model
on disk.

Raises
------
    Exception: if the model is not initialized it raises an exception

#### train

```python
def train() -> tuple[float, torch.tensor]
```

Train the network and computes loss and accuracy.

Raises
------
    Exception: Raises an exception when Federated Learning is not initialized

Returns
-------
    Tuple[float, float]: Loss and accuracy on the training set.

#### train\_with\_differential\_privacy

```python
def train_with_differential_privacy() -> tuple[float, float, float]
```

Train the network using differential privacy and computes loss and accuracy.

Raises
------
    Exception: Raises an exception when Federated Learning is not initialized

Returns
-------
    Tuple[float, float]: Loss and accuracy on the training set.

#### evaluate\_model

```python
def evaluate_model() -> tuple[float, float, float, float, float, list]
```

Validate the network on the entire test set.

Raises
------
    Exception: Raises an exception when Federated Learning is not initialized

Returns
-------
    Tuple[float, float]: loss and accuracy on the test set.

#### init\_privacy\_with\_epsilon

```python
def init_privacy_with_epsilon(phase: Phase, epsilon: float) -> None
```

Initialize differential privacy using the epsilon parameter.

**Arguments**:

- `phase` _Phase_ - phase of the training
- `EPSILON` _float_ - epsilon parameter for differential privacy
  
  Raises
  ------
- `Exception` - Preference is not initialized

#### init\_privacy\_with\_noise

```python
def init_privacy_with_noise(phase: Phase) -> None
```

Initialize differential privacy using the noise parameter
without the epsilon parameter.
Noise multiplier: the more is higher the more is the noise
Max grad: the more is higher the less private is training.

**Arguments**:

  ----
- `phase` _Phase_ - phase of the training

**Raises**:

- `Exception` - Preference is not initialized

#### init\_differential\_privacy

```python
def init_differential_privacy(
    phase: Phase
) -> tuple[nn.Module, optim.Optimizer, torch.utils.data.DataLoader]
```

Initialize the differential privacy.

**Arguments**:

- `phase` _str_ - phase of the training
  
  Returns
  -------
  _type_:

