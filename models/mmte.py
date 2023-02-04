import torch
from torch import nn


class MMTE(nn.Module):

    # d_model = 12
    def __init__(self, f1, f2, d_model, n):
        super(MMTE, self).__init__()
        self.d_model = d_model
        self.seq_len1 = 1000
        self.seq_len2 = 20

        self.lp_m1 = nn.Linear(f1, self.seq_len1 * d_model)
        self.lp_m2 = nn.Linear(f2, self.seq_len2 * d_model)

        self.cls1 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.cls2 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.positions1 = nn.Parameter(torch.rand(self.seq_len1 + 1, d_model))
        self.positions2 = nn.Parameter(torch.rand(self.seq_len2 + 1, d_model))
        # self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)

        encoder_layers1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, activation='gelu', batch_first=True)
        encoder_layers2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, activation='gelu', batch_first=True)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layers1, num_layers=6)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layers2, num_layers=6)

        # mmtemlp
        # self.decoder = nn.Sequential()
        # self.decoder.add_module('fc1', nn.Linear(d_model, 1024))
        # self.decoder.add_module('relu1', nn.ReLU(inplace=True))
        # self.decoder.add_module('fc2', nn.Linear(1024, 512))
        # self.decoder.add_module('relu2', nn.ReLU(inplace=True))
        # self.decoder.add_module('fc3', nn.Linear(512, 128))
        # self.decoder.add_module('relu3', nn.ReLU(inplace=True))
        # self.decoder.add_module('fc4', nn.Linear(128, 128))
        # self.decoder.add_module('fc5', nn.Linear(128, n))

        # mte
        self.decoder = nn.Linear(d_model, n)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        lx1 = self.lp_m1(x1)
        lx2 = self.lp_m2(x2)
        lx1 = torch.reshape(lx1, (batch_size, self.seq_len1, self.d_model))
        lx2 = torch.reshape(lx2, (batch_size, self.seq_len2, self.d_model))

        # lx1 = self.pos_encoder(lx1)
        # lx2 = self.pos_encoder(lx2)
        cls_tok1 = repeat(self.cls1, '() n e -> b n e', b=batch_size)
        cls_tok2 = repeat(self.cls2, '() n e -> b n e', b=batch_size)
        lx1 = torch.cat([cls_tok1, lx1], dim=1)
        lx2 = torch.cat([cls_tok2, lx2], dim=1)
        lx1 += self.positions1
        lx2 += self.positions2

        output1 = self.transformer_encoder1(lx1)
        output2 = self.transformer_encoder2(lx2)

        # output1 = torch.mean(output1, 1, True)
        # output2 = torch.mean(output2, 1, True)
        # decoder_in = torch.add(output1, output2)
        decoder_in = torch.add(output1[:, 0, :], output2[:, 0, :])
        output = self.decoder(decoder_in.reshape(batch_size, self.d_model))

        del lx1, lx2, output1, output2, decoder_in

        return output


class MMTE2(nn.Module):

    def __init__(self, f, d_model, n):
        super(MMTE2, self).__init__()
        self.d_model = d_model
        self.seq_len1 = 1000
        self.seq_len2 = 20

        self.lp_m1 = nn.Linear(f1, self.seq_len1 * d_model)
        self.lp_m2 = nn.Linear(f2, self.seq_len2 * d_model)

        self.cls1 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.cls2 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.positions1 = nn.Parameter(torch.rand(self.seq_len1 + 1, d_model))
        self.positions2 = nn.Parameter(torch.rand(self.seq_len2 + 1, d_model))