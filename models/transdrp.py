import math
import torch
from torch import nn, Tensor

if torch.cuda.is_available():
    my_device = torch.device('cuda:0')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')


class TransDRP(nn.Module):

    def __init__(self, m1: int, m2: int, n: int = 1, d_model: int = 4, e1: int = 8, e2: int = 8):
        super(TransDRP, self).__init__()
        self.e1_proj = nn.Linear(m1, e1).to(my_device)
        self.e2_proj = nn.Linear(m2, e2).to(my_device)

        self.gene_proj, self.desc_proj = [], []
        for i in range(e1):
            self.gene_proj.append(nn.Linear(1, d_model).to(my_device))
        for i in range(e2):
            self.desc_proj.append(nn.Linear(1, d_model).to(my_device))

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)

        gene_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, activation='gelu', batch_first=True)
        desc_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, activation='gelu', batch_first=True)
        self.gene_encoder = nn.TransformerEncoder(gene_encoder_layer, num_layers=2)
        self.desc_encoder = nn.TransformerEncoder(desc_encoder_layer, num_layers=2)

        self.head = nn.Parameter(torch.rand(1, 1, d_model))
        embed_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, activation='gelu', batch_first=True)
        self.embed_encoder = nn.TransformerEncoder(embed_encoder_layer, num_layers=2)

        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(d_model, 128))
        self.mlp.add_module('act2', nn.GELU())
        self.mlp.add_module('fc3', nn.Linear(128, n))

    def forward(self, x1: Tensor, x2: Tensor):
        batch_size = x1.shape[0]
        e1 = self.e1_proj(x1)
        e2 = self.e2_proj(x2)

        ccl_embeds, drug_embeds = (), ()
        for i in range(len(self.gene_proj)):
            # x (batch, d_model)
            x = self.gene_proj[i](e1[:, i].view(-1, 1))
            ccl_embeds = ccl_embeds + (x.view(-1, 1, x.shape[1]), )
        for i in range(len(self.desc_proj)):
            x = self.desc_proj[i](e2[:, i].view(-1, 1))
            drug_embeds = drug_embeds + (x.view(-1, 1, x.shape[1]), )
        ccl_embed = torch.cat(ccl_embeds, dim=1)
        drug_embed = torch.cat(drug_embeds, dim=1)

        ccl_encoded = self.gene_encoder(ccl_embed)
        drug_encoded = self.desc_encoder(self.pos_encoder(drug_embed))

        head = self.head.repeat(batch_size, 1, 1)
        embed = torch.cat((head, ccl_encoded, drug_encoded), dim=1)
        embed_encoded = self.embed_encoder(embed)

        output = self.mlp(embed_encoded[:, 0, :])

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
