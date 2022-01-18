from os import X_OK
import torch.nn as nn
import torch
from torch.nn.modules.activation import ReLU

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        x = self.layer(x)
        x = torch.squeeze(x)
        return x 

class LabelPredictor(nn.Module):
    def __init__(self, input_size=48 * 4 * 4, n_class=10):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, n_class)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

class DomainClassifier(nn.Module):
    def __init__(self, input_size=48 * 4 * 4):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.layer(x)
        x = self.out(x)
        return x 
