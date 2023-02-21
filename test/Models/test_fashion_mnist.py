import os
import sys

import torch
from torch import nn

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from pistacchio_simulator.Models.fashion_mnist import FashionMnistNet


class TestFashionMnistNet:
    """This class is used to test the FashionMnistNet class."""

    @staticmethod
    def test_FashionMnistNet():
        """Test for the FashionMnistNet class."""
        # test the initialization
        model = FashionMnistNet()
        assert isinstance(model, nn.Module)
        assert isinstance(model.conv, nn.Sequential)
        assert isinstance(model.fc1, nn.Sequential)

        # test the forward pass
        input_data = torch.rand(1, 1, 28, 28)
        output = model(input_data)
        assert output.shape == (1, 10)
        assert torch.max(output) <= 0.0

        # test the output of the last linear layer
        input_data = torch.rand(1, 1, 28, 28)
        output = model(input_data)
        assert output.shape == (1, 10)
        assert torch.max(output) <= 0.0

        # test the output of the last linear layer
        input_data = torch.rand(4, 1, 28, 28)
        output = model(input_data)
        assert output.shape == (4, 10)
        assert torch.max(output) <= 0.0

        # test the output of the last linear layer
        input_data = torch.rand(10, 1, 28, 28)
        output = model(input_data)
        assert output.shape == (10, 10)
        assert torch.max(output) <= 0.0
