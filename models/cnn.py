import torch
import torch.nn as nn

if torch.cuda.is_available():
    my_device = torch.device('cuda:0')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')


class CNN(nn.Module):

    def __init__(self, m1, m2, n):
        super(CNN, self).__init__()
        self.gene_proj = nn.Linear(m1, 64).to(my_device)
        self.desc_proj = nn.Linear(m2, 64).to(my_device)

        self.cnn = nn.Sequential()
        # (W-F+2P)/S+1
        self.cnn.add_module('cov1', nn.Conv2d(1, 8, (5, 5)))   # (64-5+0)/1+1=60
        self.cnn.add_module('mp1', nn.MaxPool2d(4, 4))  # (60-4+0)/4+1=15

        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(8 * 15 * 15, 16))
        self.mlp.add_module('act1', nn.ReLU(inplace=True))
        self.mlp.add_module('fc2', nn.Linear(16, n))

    def forward(self, x1, x2):
        emb1 = self.gene_proj(x1)
        emb2 = self.desc_proj(x2)
        emb = torch.matmul(emb1.view(-1, 64, 1), emb2.view(-1, 1, 64))
        output_cnn = self.cnn(emb.view(-1, 1, 64, 64))
        output = self.mlp(output_cnn.view(-1, 8 * 15 * 15))

        return output
