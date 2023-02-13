---
sidebar_label: cifar
title: cifar
---

## CifarNet Objects

```python
class CifarNet(nn.Module)
```

This class defines the network we&#x27;ll use.

#### \_\_init\_\_

```python
def __init__(in_channels: int = 3,
             num_classes: int = 10,
             dropout_rate: float = 0.3) -> None
```

Initializes the CifarNet network.

**Arguments**:

  ----
- `in_channels` _int, optional_ - number of input channels. Defaults to 3.
- `num_classes` _int, optional_ - number of classes. Defaults to 10.
- `dropout_rate` _float, optional_ - Dropout rate you want to use. Defaults to 0.3.

#### forward

```python
def forward(input_data: Tensor) -> Tensor
```

Defines the forward pass of the network.

**Arguments**:

- `input_data` _Tensor_ - Input data
  
  Returns
  -------
- `Tensor` - Output data

#### num\_flat\_features

```python
@staticmethod
def num_flat_features(input_data: Tensor) -> int
```

_summary_.

**Arguments**:

- `x` _Tensor_ - _description_
  
  Returns
  -------
- `int` - _description_

