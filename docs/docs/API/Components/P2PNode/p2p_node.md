---
sidebar_label: p2p_node
title: p2p_node
---

## P2PNode Objects

```python
class P2PNode(FederatedNode)
```

P2P Node is the component that can be used to train
the model inside a P2P network.

#### \_\_init\_\_

```python
def __init__(node_id: str, preferences: Preferences,
             logging_queue: CommunicationChannel) -> None
```

Constructor of the class.

**Arguments**:

  ----
- `node_id` _str_ - identifier of this node
- `preferences` _Preferences_ - configuration for the experiment.
- `logging_queue` _Queue_ - Queue to send the performance logs
  to the main thread.

#### set\_neighbors

```python
def set_neighbors(neighbors: list[CommunicationChannel]) -> None
```

This function is used to set the neighbors of the node.
The list passed as parameter contains all the communication channels
of the other nodes in the cluster.

**Arguments**:

- `neighbors` _List[CommunicationChannel]_ - communication
  channels of the other nodes.
  
  Raises
  ------
- `ValueError` - if the list of neighbors is empty.

#### broadcast\_weights

```python
def broadcast_weights(weights: Weights) -> None
```

This function is used to broadcast the weights to the
other nodes of the cluster.

**Arguments**:

  ----
- `weights` _Weights_ - weights to be broadcasted.

#### handle\_received\_data

```python
@staticmethod
def handle_received_data(received_data: list[Any]) -> dict[str, Any]
```

This function is used to extract the weights from the received data.

**Arguments**:

- `received_data` _List[Any]_ - list of all the data
  received from the other nodes.
  
  Returns
  -------
  Dict[str, Any]: dictionary containing the weights of the other nodes.

#### update\_weights

```python
@staticmethod
def update_weights(federated_model: FederatedModel,
                   new_weights: dict[str, Any]) -> None
```

This function is used to update the weights of the model.
It computes the average of the received weights and updates the model.

**Arguments**:

  ----
- `federated_model` _FederatedModel_ - model to be updated.
- `new_weights` _Dict[str, Any]_ - dictionary containing the
  weights of the other nodes.

#### receive\_weights\_from\_other\_peers

```python
def receive_weights_from_other_peers() -> list[Any]
```

This function is used to receive the weights from the other nodes
of the cluster.

Returns
-------
    List[Any]: Received data

#### update\_nodes\_in\_cluster

```python
def update_nodes_in_cluster(federated_model: FederatedModel) -> None
```

This function contains the logic to broadcast the weights to
the other nodes of the cluster, to receive the weights
from the other nodes and to update the model.

**Arguments**:

  ----
- `federated_model` _FederatedModel_ - model to be updated.

#### train\_in\_cluster

```python
def train_in_cluster(phase: Phase, federated_model: FederatedModel) -> dict
```

This function is called when we have to train the model inside
the cluster. It is called by each node that wants to train the model.

**Arguments**:

  ----
- `phase` __type__ - phase of the training (P2P or Server)
- `federated_model` __type__ - model we want to train

**Returns**:

- `_type_` - _description_

#### start\_p2p\_phase

```python
def start_p2p_phase(
        federated_model: FederatedModel,
        phase: Phase) -> tuple[list[float], list[float], list[float]]
```

This is the logic of the P2P Phase.
We train the model inside the cluster and then
we update the model using the weights.

**Arguments**:

- `federated_model` _FederatedModel_ - model to be trained.
- `phase` _Phase_ - phase of the training (P2P or Server)
  
  Returns
  -------
- `Dict` - List of losses, accuracies and epsilons

#### start\_node

```python
def start_node(model: nn.Module) -> None
```

Logic of the P2P Node.
This function is called to start the node, intialize the model and
start the P2P phase. After the completion of the P2P Phase then
the performances of the model are computed.

**Arguments**:

  ----
- `model` __type__ - _description_

