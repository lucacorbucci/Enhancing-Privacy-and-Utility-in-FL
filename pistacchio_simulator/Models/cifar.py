import warnings

from torch import Tensor, nn


DATA_ROOT = "../data/federated_split"
warnings.simplefilter("ignore")


class CifarNet(nn.Module):
    """This class defines the network we'll use."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initializes the CifarNet network.

        Args:
            in_channels (int, optional): number of input channels. Defaults to 3.
            num_classes (int, optional): number of classes. Defaults to 10.
            dropout_rate (float, optional): Dropout rate you want to use. Defaults to 0.3.
        """
        super().__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2
        self.layers = []
        in_dim = in_channels
        for _ in range(4):
            self.layers.append(
                nn.Conv2d(in_dim, self.out_channels, 3, self.stride, self.padding),
            )
            in_dim = self.out_channels
        self.layers = nn.ModuleList(self.layers)

        self.gn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_features = (
            self.out_channels
            * (self.stride + self.padding)
            * (self.stride + self.padding)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.ful_con = nn.Linear(num_features, num_classes)

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns
        -------
            Tensor: Output data
        """
        for conv in self.layers:
            input_data = self.gn_relu(conv(input_data))

        out = input_data.view(-1, self.num_flat_features(input_data))
        out = self.ful_con(self.dropout(out))
        return out

    @staticmethod
    def num_flat_features(input_data: Tensor) -> int:
        """_summary_.

        Args:
            x (Tensor): _description_

        Returns
        -------
            int: _description_
        """
        size = input_data.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features
