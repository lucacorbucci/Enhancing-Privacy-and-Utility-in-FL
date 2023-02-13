---
sidebar_label: celeba
title: celeba
---

## CelebaNet Objects

```python
class CelebaNet(nn.Module)
```

This class defines the CelebaNet.

#### \_\_init\_\_

```python
def __init__(in_channels: int = 3,
             num_classes: int = 10,
             dropout_rate: float = 0.2) -> None
```

Initializes the CelebaNet network.

**Arguments**:

  ----
- `in_channels` _int, optional_ - Number of input channels . Defaults to 3.
- `num_classes` _int, optional_ - Number of classes . Defaults to 2.
- `dropout_rate` _float, optional_ - _description_. Defaults to 0.2.

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

## CelebaGenderNet Objects

```python
class CelebaGenderNet(nn.Module)
```

This class defines the CelebaNet.

#### \_\_init\_\_

```python
def __init__(in_channels: int = 3,
             num_classes: int = 2,
             dropout_rate: float = 0.0) -> None
```

Initializes the CelebaNet network.

**Arguments**:

  ----
- `in_channels` _int, optional_ - Number of input channels . Defaults to 3.
- `num_classes` _int, optional_ - Number of classes . Defaults to 2.
- `dropout_rate` _float, optional_ - _description_. Defaults to 0.2.

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

