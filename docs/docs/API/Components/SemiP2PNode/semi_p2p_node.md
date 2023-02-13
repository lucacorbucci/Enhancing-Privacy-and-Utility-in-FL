---
sidebar_label: semi_p2p_node
title: semi_p2p_node
---

## SemiP2PNode Objects

```python
class SemiP2PNode(P2PNode)
```

This defines the SemiP2P node that is used to train the model
inside the P2P cluster and then with the server in a classic FL scenario.

#### \_\_init\_\_

```python
def __init__(node_id: str, preferences: Preferences,
             logging_queue: CommunicationChannel) -> None
```

Initialize the SemiP2PNode.

**Arguments**:

  ----
- `node_id` _str_ - id of the node
- `preferences` _Preferences_ - configuration file of the node
- `logging_queue` _CommunicationChannel_ - queue that is used to send back the
  performances of the node to the main thread.

#### local\_evaluation

```python
@staticmethod
def local_evaluation(federated_model: FederatedModel) -> dict
```

This function evaluates the performances of the model
on the local test set.

**Arguments**:

- `federated_model` _FederatedModel_ - the model to evaluate
  
  Returns
  -------
- `dict` - a dictionary containing the performances of the model

#### mixed\_training

```python
def mixed_training(
        federated_model: FederatedModel,
        model: nn.Module) -> tuple[list[float], list[float], list[float]]
```

This function implements the mixed training.
It alternates between training with the server and training
inside the P2P cluster.

**Arguments**:

  ----
- `federated_model` _FederatedModel_ - federated model to train
- `model` _nn.Module_ - model we use in the FederatedModel

**Returns**:

  Tuple[List[float], List[float], List[float]]: loss, accuracy,
  epsilon of the trained model

#### start\_node

```python
def start_node(model: nn.Module) -> None
```

Start the semi p2p node. This function implements the logic of the
semi p2p node, first of all the node trains the model inside the cluster
and then exchanges the weights with the server. Then it starts the training
with the server. If mixed mode is enabled, it alternates between training
inside the cluster and training with the server.

**Arguments**:

  ----
- `model` __type__ - Model that we want to use during the federated learning

