import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(512*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout) # 0.5 was the suitable one
        self.log_softmax = nn.LogSoftmax()

    # F.relu ? nn.ReLU() , difference ???
    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = self.bn3(self.pool(F.relu(self.conv3(x))))
        x = self.bn4(self.pool(F.relu(self.conv4(x))))
        x = self.bn5(self.pool(F.relu(self.conv5(x))))


        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    #     self.backbone = nn.Sequential(
    #         # Block 1
    #         nn.Conv2d(3, 16, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(2, 2),
    #         nn.BatchNorm2d(16),

    #         # Block 2
    #         nn.Conv2d(16, 64, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(2, 2),
    #         nn.BatchNorm2d(64),

    #         # Block 3
    #         nn.Conv2d(64, 128, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(2, 2),
    #         nn.BatchNorm2d(128),

    #         # Block 4
    #         nn.Conv2d(128, 256, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(2, 2),
    #         nn.BatchNorm2d(256),

    #         # Block 5
    #         nn.Conv2d(256, 512, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(2, 2),
    #         nn.BatchNorm2d(512),
    #     )
    #     self.head = nn.Sequential(
    #         nn.Dropout1d(dropout, inplace=False),
    #         nn.Linear(7 * 7 * 512, 128, bias=True),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout1d(dropout, inplace=False),
    #         nn.Linear(128, num_classes, bias=True),
    #     )  # Output size: N x num_classes

    #     # # Add weights initialisation
    #     # for m in self.modules():
    #     #     if isinstance(m, nn.Conv2d):
    #     #         I.xavier_normal_(m.weight)
    #     #         if m.bias is not None:
    #     #             I.constant_(m.bias, 0)
    #     #     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
    #     #         I.constant_(m.weight, 1)
    #     #         I.constant_(m.bias, 0)
    #     #     elif isinstance(m, nn.Linear):
    #     #         I.xavier_normal_(m.weight)
    #     #         I.constant_(m.bias, 0)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.backbone(x)
    #     x = torch.flatten(x, 1)
    #     x = self.head(x)
    #     return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
