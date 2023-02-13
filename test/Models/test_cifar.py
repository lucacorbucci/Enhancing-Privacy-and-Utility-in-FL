import os
import sys

import torch

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from pistacchio.Models.cifar import CifarNet


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

    @staticmethod
    def test_cifar_net_dropout_rate():
        """Test that the CifarNet applies the correct dropout rate."""
        net = CifarNet(dropout_rate=0.5)
        input_data = torch.randn(1, 3, 32, 32)
        net(input_data)
        assert net.dropout.p == 0.5

    @staticmethod
    def test_cifar_net_num_flat_features():
        """Test that the num_flat_features function returns correct number of features."""
        net = CifarNet()
        input_data = torch.randn(1, 3, 32, 32)
        num_features = net.num_flat_features(input_data)
        assert num_features == 32 * 32 * 3
