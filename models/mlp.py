import torch
import torch.nn as nn

if torch.cuda.is_available():
    my_device = torch.device('cuda:0')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')


class MLP(nn.Module):

    def __init__(self, m1, m2, n):
        super().__init__()
        self.regression = nn.Sequential()
        self.regression.add_module('fc1', nn.Linear(m1 + m2, 1024))
        # self.regression.add_module('bn1', nn.BatchNorm1d(1024))
        self.regression.add_module('relu1', nn.ReLU(inplace=True))
        self.regression.add_module('do1', nn.Dropout(p=0.2))
        self.regression.add_module('fc2', nn.Linear(1024, 1024))
        # self.regression.add_module('bn2', nn.BatchNorm1d(1024))
        self.regression.add_module('relu2', nn.ReLU(inplace=True))
        self.regression.add_module('do2', nn.Dropout(p=0.2))
        self.regression.add_module('fc3', nn.Linear(1024, 1024))
        self.regression.add_module('relu3', nn.ReLU(inplace=True))
        self.regression.add_module('fc4', nn.Linear(1024, 1024))
        self.regression.add_module('relu4', nn.ReLU(inplace=True))
        self.regression.add_module('fc5', nn.Linear(1024, n))

    def forward(self, x1, x2):

        return self.regression(torch.cat((x1, x2), dim=1))


class ResMLP50(nn.Module):

    def __init__(self, m1, m2, n):
        super().__init__()

        n_neuron = 512
        self.blk_qty = 24

        self.input_layer = nn.Sequential()
        self.input_layer.add_module('input_fc', nn.Linear(m1 + m2, n_neuron))
        self.input_layer.add_module('input_bn', nn.BatchNorm1d(n_neuron))
        self.output_layer = nn.Linear(n_neuron, n)

        self.res_blk_list = []
        for i in range(self.blk_qty):
            res_blk = nn.Sequential()
            res_blk.add_module('resblk{}_fc0'.format(i), nn.Linear(n_neuron, n_neuron))
            res_blk.add_module('resblk{}_bn0'.format(i), nn.BatchNorm1d(n_neuron))
            res_blk.add_module('resblk{}_relu0'.format(i), nn.GELU())
            res_blk.add_module('resblk{}_do0'.format(i), nn.Dropout(0.1))
            res_blk.add_module('resblk{}_fc1'.format(i), nn.Linear(n_neuron, n_neuron))
            res_blk.add_module('resblk{}_bn1'.format(i), nn.BatchNorm1d(n_neuron))
            res_blk.add_module('resblk{}_relu1'.format(i), nn.GELU())
            res_blk.add_module('resblk{}_do1'.format(i), nn.Dropout(0.1))
            self.res_blk_list.append(res_blk)

    def forward(self, x1, x2):
        data_in = torch.cat((x1, x2), dim=1)
        # data_in_relu = nn.ReLU().to(my_device)
        res_in = self.input_layer(data_in)
        for i in range(self.blk_qty):
            current_blk = self.res_blk_list[i].to(my_device)
            current_out = current_blk(res_in)
            current_out_2 = torch.clone(current_out)
            current_out_2 += res_in
            res_in = torch.clone(current_out_2)
        output = self.output_layer(res_in)

        return output



