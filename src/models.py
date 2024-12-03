import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    """Class for a fully connected neural network."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass step.

        Args:
            x (torch.Tensor): Input data to the network.

        Returns:
            torch.Tensor: Output data to the network.
        """
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Lenet5(nn.Module):
    """Class for the LeNet5 neural network."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=(5, 5), padding=(2, 2)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dense1 = nn.Linear(in_features=400, out_features=120)
        self.dense2 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass step.

        Args:
            x (torch.Tensor): Input data to the network.

        Returns:
            torch.Tensor: Output data to the network.
        """
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = F.relu(self.pool2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.output(x)
        return x
