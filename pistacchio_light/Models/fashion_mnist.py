import torch
import torch.nn.functional as F
from torch import Tensor, nn

torch.manual_seed(42)


class FashionMnistNet(nn.Module):
    """Fashion Mnist network definition."""

    def __init__(self) -> None:
        """Initialization of the network."""
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 12 * 12, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns
        -------
            Tensor: Output data
        """
        out = self.conv(input_data)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 64 * 12 * 12)
        out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return out
