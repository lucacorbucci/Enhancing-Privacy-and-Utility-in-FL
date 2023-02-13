---
sidebar_label: federated_node
title: federated_node
---

## FederatedNode Objects

```python
class FederatedNode()
```

FederatedNode is the component that can be used for the
classic Federated Learning.
It trains the model locally and then sends the weights to the server.

#### \_\_init\_\_

```python
def __init__(node_id: str, preferences: Preferences,
             logging_queue: CommunicationChannel) -> None
```

Init the Federated Node.

**Arguments**:

  ----
- `node_id` _str_ - id of the node
- `preferences` _Preferences_ - preferences object of the node that contains
  all the preferences for this node
- `logging_queue` _CommunicationChannel_ - queue that is used to send back the
  performances of the node to the main thread.

#### receive\_data\_from\_server

```python
def receive_data_from_server() -> Any
```

This function receives the weights from the server.
If the weights are not received, it returns an error message
otherwise it returns the weights.

Returns
-------
    Union[Weights, None]: Weights received from the server

#### send\_weights\_to\_server

```python
def send_weights_to_server(weights: Weights) -> None
```

This function is used to send the weights of the nodes to the server.

**Arguments**:

  ----
- `weights` _Weights_ - weights to be sent to the server

**Raises**:

- `ValueError` - Raised when the server channel is not initialized

#### add\_server\_channel

```python
def add_server_channel(server_channel: CommunicationChannel) -> None
```

This function adds the server channel to the sender thread.

**Arguments**:

  ----
- `server_channel` __type__ - server channel

#### init\_federated\_model

```python
def init_federated_model(model: nn.Module) -> FederatedModel
```

Initialize the federated learning model.

**Arguments**:

- `model` __type__ - _description_
  
  Returns
  -------
- `FederatedModel` - _description_

#### get\_communication\_channel

```python
def get_communication_channel() -> CommunicationChannel
```

Getter for the communication channel of this node.

Returns
-------
    CommunicationChannel: _description_

#### local\_training

```python
@staticmethod
def local_training(differential_private_train: bool,
                   federated_model: FederatedModel) -> dict
```

_summary_.

**Arguments**:

- `differential_private_train` _bool_ - _description_
- `federated_model` _FederatedModel_ - _description_
  
  Returns
  -------
- `dict` - _description_

#### send\_and\_receive\_weights\_with\_server

```python
def send_and_receive_weights_with_server(federated_model: FederatedModel,
                                         metrics: dict,
                                         results: dict | None = None) -> Any
```

Send weights to the server and receive the
updated weights from the server.

**Arguments**:

- `federated_model` _FederatedModel_ - Federated model
- `metrics` _dict_ - metrics computed on the node (loss, accuracy, epsilon)
  
  Returns
  -------
- `_type_` - weights received from the server

#### start\_server\_phase

```python
def start_server_phase(
    federated_model: FederatedModel,
    results: dict | None = None
) -> tuple[list[float], list[float], list[float]]
```

This function starts the server phase of the federated learning.
In particular, it trains the model locally and then sends the weights.
Then the updated weights are received and used to update
the local model.

**Arguments**:

- `federated_model` _FederatedModel_ - _description_
  
  Returns
  -------
  Tuple[List[float], List[float], List[float]]: _description_

#### send\_performances

```python
def send_performances(performances: dict[str, Performances]) -> None
```

This function is used to send the performances of
the node to the server.

**Arguments**:

  ----
- `performances` _Performances_ - _description_

#### compute\_performances

```python
def compute_performances(loss_list: list, accuracy_list: list, phase: str,
                         message_counter: int,
                         epsilon_list: list | None) -> dict
```

This function is used to compute the performances
of the node. In particulare we conside the list of
loss, accuracy and epsilon computed during the
local training on the node.

**Arguments**:

- `loss_list` _List_ - list of loss computed during the local training
- `accuracy_list` _List_ - list of accuracy computed during the local training
- `phase` _str_ - Phase of the training (P2P or server)
- `message_counter` _int_ - count of the exchanged messages
- `epsilon_list` _List, optional_ - list of epsilon computed
  during the local training. Defaults to None.
  
  Returns
  -------
- `Performances` - Performance object of the node

#### receive\_starting\_model\_from\_server

```python
def receive_starting_model_from_server(
        federated_model: FederatedModel) -> None
```

This function is used to receive the starting model
from the server so that all the nodes start the federated training
from the same random weights.

**Arguments**:

  ----
- `federated_model` _FederatedModel_ - The federated model we want
  to initialize with the received weights

#### start\_node

```python
def start_node(model: nn.Module) -> None
```

This method implements all the logic of the federated node.
It starts the training of the model and then sends the weights to the
server.
Then, after the end of the training, it sends the performances of the
node to the main thread.

**Arguments**:

  ----
- `model` __type__ - Model that we want to use during the federated learning

