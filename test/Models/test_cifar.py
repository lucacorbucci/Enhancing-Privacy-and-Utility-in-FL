import os
import sys

import torch

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from pistacchio_simulator.Models.cifar import CifarNet


class TestCifarNet:
    """This class is used to test the CifarNet class."""

    @staticmethod
    def test_cifar_net_output_shape():
        """Test that the output of the CifarNet has the correct shape."""
        net = CifarNet()
        input_data = torch.randn(1, 3, 32, 32)
        output_data = net(input_data)

        assert output_data.shape == (1, 10)

    @staticmethod
    def test_cifar_net_batch_size():
        """Test that the CifarNet can handle different batch sizes."""
        net = CifarNet()
        input_data = torch.randn(32, 3, 32, 32)
        output_data = net(input_data)

        assert output_data.shape == (32, 10)

    @staticmethod
    def test_cifar_net_multiple_channels():
        """Test that the CifarNet can handle multiple channels."""
        net = CifarNet()
        input_data = torch.randn(1, 3, 32, 32)
        output_data = net(input_data)

        assert output_data.shape == (1, 10)
