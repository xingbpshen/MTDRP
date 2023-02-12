import torch
import torch.nn as nn


class GradientReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb

        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        new_grad = gradient.neg() * ctx.lamb

        return new_grad, None


class DANN(nn.Module):

    def __init__(self, m1, m2, n=2):
        super(DANN, self).__init__()
        self.Gf = torch.nn.Sequential()
        self.Gf.add_module('gf_fc1', nn.Linear(m1 + m2, 1024))
        self.Gf.add_module('gf_act1', nn.ReLU(inplace=True))
        self.Gf.add_module('gf_do1', nn.Dropout(p=0.2))
        self.Gf.add_module('gf_fc2', nn.Linear(1024, 1024))
        self.Gf.add_module('gf_act2', nn.ReLU(inplace=True))

        self.Gy = torch.nn.Sequential()
        self.Gy.add_module('gy_fc1', nn.Linear(1024, 1024))
        self.Gy.add_module('gy_act1', nn.ReLU(inplace=True))
        self.Gy.add_module('gy_do1', nn.Dropout(p=0.2))
        self.Gy.add_module('gy_fc2', nn.Linear(1024, 1024))
        self.Gy.add_module('gy_act2', nn.ReLU(inplace=True))
        self.Gy.add_module('gy_fc3', nn.Linear(1024, n))
        self.Gy.add_module('gy_softmax', nn.LogSoftmax(dim=1))

        self.Gd = torch.nn.Sequential()
        self.Gd.add_module('gd_fc1', nn.Linear(1024, 1024))
        self.Gd.add_module('gd_act1', nn.ReLU(inplace=True))
        self.Gy.add_module('gd_do1', nn.Dropout(p=0.2))
        self.Gd.add_module('gd_fc2', nn.Linear(1024, 1024))
        self.Gd.add_module('gd_act2', nn.ReLU(inplace=True))
        self.Gd.add_module('gd_fc3', nn.Linear(1024, 2))
        self.Gd.add_module('gd_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x1, x2, lamb):
        feature = self.Gf(torch.cat((x1, x2), dim=1))
        reverse = GradientReverse.apply(feature, lamb)
        cate = self.Gy(feature)
        domain_classification = self.Gd(reverse)

        return cate, domain_classification
