import os
import sys

import torch

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from pistacchio.Models.celeba import CelebaGenderNet, CelebaNet


class TestCelebaNet:
    """This class is used to test the CelebaNet class."""

    @staticmethod
    def test_celeba_net_output_shape():
        """Test that the output of the CelebaNet has the correct shape."""
        net = CelebaNet()
        input_data = torch.randn(1, 3, 64, 64)
        output_data = net(input_data)

        assert output_data.shape == (1, 10)

    @staticmethod
    def test_celeba_net_batch_size():
        """Test that the CelebaNet can handle different batch sizes."""
        net = CelebaNet()
        input_data = torch.randn(32, 3, 64, 64)
        output_data = net(input_data)

        assert output_data.shape == (32, 10)

    @staticmethod
    def test_celeba_net_multiple_channels():
        """Test that the CelebaNet can handle multiple channels."""
        net = CelebaNet()
        input_data = torch.randn(1, 3, 64, 64)
        output_data = net(input_data)

        assert output_data.shape == (1, 10)

    @staticmethod
    def test_celeba_gender_net_output_shape():
        """Test that the output of the CelebaGenderNet has the correct shape."""
        net = CelebaGenderNet()
        input_data = torch.randn(1, 3, 64, 64)
        output_data = net(input_data)

        assert output_data.shape == (1, 2)

    @staticmethod
    def test_celeba_gender_net_batch_size():
        """Test that the CelebaGenderNet can handle different batch sizes."""
        net = CelebaGenderNet()
        input_data = torch.randn(32, 3, 64, 64)
        output_data = net(input_data)

        assert output_data.shape == (32, 2)

    @staticmethod
    def test_celeba_gender_net_multiple_channels():
        """Test that the CelebaGenderNet can handle multiple channels."""
        net = CelebaGenderNet()
        input_data = torch.randn(1, 3, 64, 64)
        output_data = net(input_data)

        assert output_data.shape == (1, 2)
