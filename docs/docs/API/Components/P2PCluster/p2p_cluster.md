---
sidebar_label: p2p_cluster
title: p2p_cluster
---

#### launch\_semi\_p2p\_node

```python
def launch_semi_p2p_node(node: SemiP2PNode,
                         nodes_queue: list[CommunicationChannel],
                         server_queue: CommunicationChannel,
                         model: FederatedModel) -> None
```

Function used to launch a SemiP2PNode.

**Arguments**:

  ----
- `node` _SemiP2PNode_ - node that will be launched
- `nodes_queue` _List[CommunicationChannel]_ - communication channel
  of the other nodes of the cluster
- `server_queue` _CommunicationChannel_ - communication channel of the server
- `model` _FederatedModel_ - model to be trained

#### launch\_p2p\_node

```python
def launch_p2p_node(node: P2PNode, nodes_queue: list[CommunicationChannel],
                    model: FederatedModel) -> None
```

Function used to launch a P2PNode.

**Arguments**:

  ----
- `node` _P2PNode_ - node that will be launched
- `nodes_queue` _List[CommunicationChannel]_ - communication channel
  of the other nodes of the cluster
- `model` _FederatedModel_ - model to be trained

## P2PCluster Objects

```python
class P2PCluster()
```

This class defines a cluster of nodes.
We will use the nodes in this cluster to perform federated learning
and to train a model.

#### \_\_init\_\_

```python
def __init__(preferences: Preferences, cluster_id: int,
             model: nn.Module) -> None
```

Constructor of the P2PCluster class.

**Arguments**:

  ----
- `preferences` _Preferences_ - configuration of the experiment
- `cluster_id` _int_ - ID of the cluster
- `model` _nn.Module_ - model to be trained

#### init\_cluster

```python
def init_cluster() -> None
```

This function starts the nodes of the cluster.
It creates a number of nodes equal to the number of nodes
specificed in the constructor and start them.

#### start\_cluster\_p2p

```python
def start_cluster_p2p() -> None
```

This function starts all the nodes of the cluster.

#### start\_cluster

```python
def start_cluster(server_queue: CommunicationChannel) -> None
```

This function starts all the nodes of the cluster.

#### get\_nodes\_receiver\_channel

```python
def get_nodes_receiver_channel() -> None
```

This function gets the receiver channel of each node.

Raises
------
    Exception: if the the nodes are not initialized.

#### get\_channels

```python
def get_channels() -> list[CommunicationChannel]
```

This function returns the channels of the nodes of the cluster.

Returns
-------
    List[CommunicationChannel]: list of the channels of the nodes of the cluster

#### stop\_cluster

```python
def stop_cluster() -> None
```

This function stops the nodes of the cluster and extract the
performances of each node.

#### set\_neighbors

```python
def set_neighbors() -> None
```

This function sets up the neighbors of each node.
In particular, for each node, a queue list containing all the
receiving queue of the other nodes is assigned.

