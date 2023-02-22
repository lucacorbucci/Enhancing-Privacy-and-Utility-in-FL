import os
import sys

import torch

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from pistacchio_simulator.Models.mnist import MnistNet


class TestMnistNet:
    """This class is used to test the MnistNet class."""

    @staticmethod
    def test_mnist_net_output_shape():
        """Test that the output of the MnistNet has the correct shape."""
        net = MnistNet()
        input_data = torch.randn(1, 1, 28, 28)
        output_data = net(input_data)

        assert output_data.shape == (1, 10)

    @staticmethod
    def test_mnist_net_batch_size():
        """Test that the MnistNet can handle different batch sizes."""
        net = MnistNet()
        input_data = torch.randn(32, 1, 28, 28)
        output_data = net(input_data)

        assert output_data.shape == (32, 10)
