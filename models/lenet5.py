import torch
import torch.nn as nn

from parch import ReLU1

class LeNet5(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10, bias=bias)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class LeNet5ReLU6(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2, bias=bias),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, bias=bias),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120, bias=bias),
            nn.ReLU6(inplace=True),
            nn.Linear(120, 84, bias=bias),
            nn.ReLU6(inplace=True),
            nn.Linear(84, 10, bias=bias),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class LeNet5ReLU1(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2, bias=bias),
            ReLU1(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, bias=bias),
            ReLU1(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120, bias=bias),
            ReLU1(inplace=True),
            nn.Linear(120, 84, bias=bias),
            ReLU1(inplace=True),
            nn.Linear(84, 10, bias=bias),
            ReLU1(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
