---
sidebar_label: adults
title: adults
---

## Adult_FederatedLearning Objects

```python
class Adult_FederatedLearning(FederatedModel)
```

#### \_\_init\_\_

```python
def __init__(node_name: str, parameters: dict = None) -> None
```

Initialize the Mnist federated learning

**Arguments**:

- `id` _int_ - node id
- `num_communication_round_pre_training` _int_ - number of epochs to train

#### create_net

```python
def create_net() -> Net
```

This function creates the network we will use

**Arguments**:

- `device` _torch.device_ - device to use

**Returns**:

- `Net` - the network we will use

#### train

```python
def train(current_epoch: int) -> Tuple[float, float]
```

Train the network and computes loss and accuracy.

**Returns**:

Tuple[float, float]: Loss and accuracy on the training set.

#### train_with_differential_privacy

```python
def train_with_differential_privacy(phase: str) -> Tuple[float, float, float]
```

Train the network using differential privacy and computes loss and accuracy.

**Arguments**:

- `net` _Net_ - the network to train
- `trainloader` _torch.utils.data.DataLoader_ - the training set
- `device` _torch.device_ - the device to use to train the network

**Returns**:

Tuple[float, float, float]: Loss, accuracy and epsilon on the training set.

#### evaluate_model

```python
def evaluate_model() -> Tuple[float, float, float, float, float]
```

Validate the network on the entire test set.

**Arguments**:

- `net` _Net_ - network to test
- `testloader` _torch.utils.data.DataLoader_ - test set
- `device` _torch.device_ - device to use to test the network

**Returns**:

Tuple[float, float]: loss and accuracy on the test set.

## AdultDataLoader Objects

```python
class AdultDataLoader()
```

#### load_splitted_dataset_train

```python
def load_splitted_dataset_train(ID: int) -> torch.utils.data.DataLoader["Any"]
```

This function loads the splitted train dataset.

**Arguments**:

- `ID` _int_ - node id

**Returns**:

- `_type_` - _description_

#### load_splitted_dataset_test

```python
def load_splitted_dataset_test(ID: int) -> torch.utils.data.DataLoader["Any"]
```

This function loads the splitted test dataset.

**Arguments**:

- `ID` _int_ - node id

**Returns**:

- `_type_` - _description_
