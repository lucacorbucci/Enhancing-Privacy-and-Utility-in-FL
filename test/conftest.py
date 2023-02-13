import random as rand

import numpy
import pytest
import torch


@pytest.fixture(autouse=True)
def random() -> None:
    """Set the random seed to 0 for all tests."""
    rand.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
