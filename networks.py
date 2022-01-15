import torch
import torch.nn as nn
import numpy as np
from transformer import *

__all__ = ['TransformerMLC']


class TransformerMLC(nn.Module):
    def __init__(self, num_features, num_labels, d_model=512, nhead=8, num_enc_layers=6, num_dec_layers=6,
                 dim_feedforward=2048, dropout=0.1, batch_first=True, bias=True, padding_idx=0, device='cpu', **kwargs):
        super(TransformerMLC, self).__init__()

        self.num_features = num_features
        self.num_labels = num_labels
        self.label_array = torch.from_numpy(np.arange(self.num_labels)).to(device)

        self.input_emb = nn.Embedding(num_features, d_model, padding_idx=padding_idx)
        enc_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                            dropout=dropout, batch_first=batch_first)
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        self.label_emb = nn.Embedding(num_labels, d_model, padding_idx=padding_idx)
        dec_layer = ModifiedDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=batch_first)
        self.decoder = TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        self.label_prj = nn.Linear(d_model, num_labels, bias=bias)

    def forward(self, features):
        print(features.device)
        enc_input = self.input_emb(features)
        input_mask = features == 0
        print(input_mask.device)
        print(self.label_array.device)
        enc_output = self.encoder(enc_input, src_key_padding_mask=input_mask)
        dec_input = self.label_emb(self.label_array).repeat(features.size(0), 1, 1)
        dec_output = self.decoder(dec_input, enc_output, memory_key_padding_mask=input_mask)

        seq_logit = self.label_prj(dec_output)
        seq_logit = torch.diagonal(seq_logit, 0, 1, 2)

        return seq_logit

