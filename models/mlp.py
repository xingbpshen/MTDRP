import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, m1, m2, n):
        super().__init__()
        self.regression = nn.Sequential()
        self.regression.add_module('fc1', nn.Linear(m1 + m2, 1024))
        self.regression.add_module('bn1', nn.BatchNorm1d(1024))
        self.regression.add_module('relu1', nn.ReLU(inplace=True))
        self.regression.add_module('do1', nn.Dropout(p=0.2))
        self.regression.add_module('fc2', nn.Linear(1024, 1024))
        self.regression.add_module('bn2', nn.BatchNorm1d(1024))
        self.regression.add_module('relu2', nn.ReLU(inplace=True))
        self.regression.add_module('do2', nn.Dropout(p=0.2))
        self.regression.add_module('fc3', nn.Linear(1024, 1024))
        self.regression.add_module('relu3', nn.ReLU(inplace=True))
        self.regression.add_module('fc4', nn.Linear(1024, 1024))
        self.regression.add_module('relu4', nn.ReLU(inplace=True))
        self.regression.add_module('fc5', nn.Linear(1024, n))

    def forward(self, x1, x2):

        return self.regression(torch.cat((x1, x2), dim=1))
