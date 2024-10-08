import torch
import torch.nn.functional as F
from torch import Tensor, nn

torch.manual_seed(42)


class MnistNet(nn.Module):
    """Mnist network definition."""

    def __init__(self) -> None:
        """Initialization of the network."""
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns
        -------
            Tensor: Output data
        """
        out = input_data.view(-1, 28 * 28)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)
