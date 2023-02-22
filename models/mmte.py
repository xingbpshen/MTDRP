import math
import torch
from torch import nn, Tensor

if torch.cuda.is_available():
    my_device = torch.device('cuda:0')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')


class MTDRP(nn.Module):

    def __init__(self, m1, m2, n=1, d_model=4):
        super(MTDRP, self).__init__()
        e1, e2 = 8, 4
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
        self.mlp.add_module('fc1', nn.Linear(d_model, d_model))
        self.mlp.add_module('act1', nn.GELU())
        self.mlp.add_module('fc2', nn.Linear(d_model, n))

    def forward(self, x1, x2):
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


# class MMTE(nn.Module):
#
#     # d_model = 12
#     def __init__(self, f1, f2, d_model, n):
#         super(MMTE, self).__init__()
#         self.d_model = d_model
#         self.seq_len1 = 1000
#         self.seq_len2 = 20
#
#         self.lp_m1 = nn.Linear(f1, self.seq_len1 * d_model)
#         self.lp_m2 = nn.Linear(f2, self.seq_len2 * d_model)
#
#         self.cls1 = nn.Parameter(torch.rand(1, 1, self.d_model))
#         self.cls2 = nn.Parameter(torch.rand(1, 1, self.d_model))
#         self.positions1 = nn.Parameter(torch.rand(self.seq_len1 + 1, d_model))
#         self.positions2 = nn.Parameter(torch.rand(self.seq_len2 + 1, d_model))
#         # self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)
#
#         encoder_layers1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, activation='gelu', batch_first=True)
#         encoder_layers2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, activation='gelu', batch_first=True)
#         self.transformer_encoder1 = nn.TransformerEncoder(encoder_layers1, num_layers=6)
#         self.transformer_encoder2 = nn.TransformerEncoder(encoder_layers2, num_layers=6)
#
#         # mmtemlp
#         # self.decoder = nn.Sequential()
#         # self.decoder.add_module('fc1', nn.Linear(d_model, 1024))
#         # self.decoder.add_module('relu1', nn.ReLU(inplace=True))
#         # self.decoder.add_module('fc2', nn.Linear(1024, 512))
#         # self.decoder.add_module('relu2', nn.ReLU(inplace=True))
#         # self.decoder.add_module('fc3', nn.Linear(512, 128))
#         # self.decoder.add_module('relu3', nn.ReLU(inplace=True))
#         # self.decoder.add_module('fc4', nn.Linear(128, 128))
#         # self.decoder.add_module('fc5', nn.Linear(128, n))
#
#         # mte
#         self.decoder = nn.Linear(d_model, n)
#
#     def forward(self, x1, x2):
#         batch_size = x1.shape[0]
#         lx1 = self.lp_m1(x1)
#         lx2 = self.lp_m2(x2)
#         lx1 = torch.reshape(lx1, (batch_size, self.seq_len1, self.d_model))
#         lx2 = torch.reshape(lx2, (batch_size, self.seq_len2, self.d_model))
#
#         # lx1 = self.pos_encoder(lx1)
#         # lx2 = self.pos_encoder(lx2)
#         cls_tok1 = repeat(self.cls1, '() n e -> b n e', b=batch_size)
#         cls_tok2 = repeat(self.cls2, '() n e -> b n e', b=batch_size)
#         lx1 = torch.cat([cls_tok1, lx1], dim=1)
#         lx2 = torch.cat([cls_tok2, lx2], dim=1)
#         lx1 += self.positions1
#         lx2 += self.positions2
#
#         output1 = self.transformer_encoder1(lx1)
#         output2 = self.transformer_encoder2(lx2)
#
#         # output1 = torch.mean(output1, 1, True)
#         # output2 = torch.mean(output2, 1, True)
#         # decoder_in = torch.add(output1, output2)
#         decoder_in = torch.add(output1[:, 0, :], output2[:, 0, :])
#         output = self.decoder(decoder_in.reshape(batch_size, self.d_model))
#
#         del lx1, lx2, output1, output2, decoder_in
#
#         return output
#
#
# class MMTE2(nn.Module):
#
#     def __init__(self, f, d_model, n):
#         super(MMTE2, self).__init__()
#         self.d_model = d_model
#         self.seq_len1 = 1000
#         self.seq_len2 = 20
#
#         self.lp_m1 = nn.Linear(f1, self.seq_len1 * d_model)
#         self.lp_m2 = nn.Linear(f2, self.seq_len2 * d_model)
#
#         self.cls1 = nn.Parameter(torch.rand(1, 1, self.d_model))
#         self.cls2 = nn.Parameter(torch.rand(1, 1, self.d_model))
#         self.positions1 = nn.Parameter(torch.rand(self.seq_len1 + 1, d_model))
#         self.positions2 = nn.Parameter(torch.rand(self.seq_len2 + 1, d_model))