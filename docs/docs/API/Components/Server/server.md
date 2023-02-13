---
sidebar_label: server
title: server
---

## Server Objects

```python
class Server()
```

This class defines the server of the federated learning model.

**Arguments**:

  ----
- `threading` __type__ - _description_

#### \_\_init\_\_

```python
def __init__(node_queues: list[CommunicationChannel], preferences: Preferences,
             model: nn.Module) -> None
```

This function initialize the server.

**Arguments**:

  ----
- `node_queues` _List[CommunicationChannel]_ - queus of
  the nodes that will send data to the server
- `preferences` _Preferences_ - configuration file of the experiment

#### collect\_updates

```python
def collect_updates() -> tuple[dict, list, dict]
```

This function is used to collect the updates from the nodes.

Returns
-------
    Tuple[Dict[str, TDestination], List[float]]: A tuple with a
    dictionary with sender_id:weights
    and a list with the epsilon of each node (when we use differential privacy)

#### send\_updates

```python
def send_updates(weights: TDestination) -> None
```

This function sends the averaged weights to the nodes.

**Arguments**:

  ----
- `weights` _TDestination_ - the averaged weights that have to be sent

#### stop\_server

```python
def stop_server() -> None
```

This function can be called to stop the server.

#### get\_queue

```python
def get_queue() -> CommunicationChannel
```

This function returns the queue of the server.

Returns
-------
    queue.Queue: the queue of the server that will be used to receive data

#### check\_improvement

```python
def check_improvement(accuracy: float) -> bool
```

This function checks if the accuracy is improving.
If it doesn&#x27;t improve for max_iterations then it will return True
else it returns False.

#### send\_stop\_signal

```python
def send_stop_signal() -> None
```

This function sends the stop signal to the nodes.

#### create\_model

```python
def create_model() -> None
```

This function creates and initialize the model that
we&#x27;ll use on the server for the validation.

#### load\_validation\_data

```python
def load_validation_data() -> None
```

This function loads the validation data from disk.

#### broadcast\_initial\_weights

```python
def broadcast_initial_weights() -> None
```

This function sends the averaged weights to the nodes.

#### save\_model

```python
def save_model(wandb_run: ModuleType) -> None
```

This function stores the model on disk.

**Arguments**:

  ----
- `wandb_run` _Run_ - wandb reference

#### write\_results

```python
def write_results(accuracy_validation: list[float]) -> None
```

This function writes on disk the results of the experiment.

#### receive\_performances\_from\_nodes

```python
def receive_performances_from_nodes() -> tuple[int, int]
```

This function is used to receive the performances from the nodes.

#### check\_requires\_grad

```python
def check_requires_grad() -> None
```

This function checks if the model requires grad or not.
If we are using fine-tuning then we don&#x27;t want to update all the weights and we want to
set requires_grad only to the classifier weights.

#### get\_server\_iterations

```python
def get_server_iterations() -> int
```

This function returns the number of server iterations.

Returns
-------
    _type_: number of server iterations

#### start\_server

```python
def start_server() -> None
```

This function is the run function of the server.
It receives the weights from the nodes and computes the average of the weights.
Then it sends back the weights to the nodes.

