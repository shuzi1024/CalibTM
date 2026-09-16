"""Small Torch-only equivalents of SPIN's pinned TSL/PyG primitives.

Parameter names and mathematical definitions follow torch_spatiotemporal 0.1.1.
The unusual std+eps normalization is deliberately NOT replaced by torch LayerNorm.
"""

import math

import torch
from torch import nn


class Dense(nn.Module):
    def __init__(self, input_size, output_size, activation="linear", dropout=0., bias=True):
        super().__init__()
        activation_cls = {"relu": nn.ReLU, "prelu": nn.PReLU,
                          "linear": nn.Identity}[activation]
        self.layer = nn.Sequential(nn.Linear(input_size, output_size, bias=bias),
                                   activation_cls(),
                                   nn.Dropout(dropout) if dropout > 0 else nn.Identity())

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, exog_size=None,
                 n_layers=1, activation="relu", dropout=0.):
        super().__init__()
        if exog_size is not None:
            input_size += exog_size
        self.mlp = nn.Sequential(*[
            Dense(input_size if i == 0 else hidden_size, hidden_size,
                  activation, dropout) for i in range(n_layers)])
        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter("readout", None)

    def forward(self, x, u=None):
        if u is not None:
            if u.ndim == 3 and x.ndim == 4:
                u = u.unsqueeze(-2).expand(*x.shape[:-1], u.shape[-1])
            x = torch.cat((x, u), dim=-1)
        out = self.mlp(x)
        return self.readout(out) if self.readout is not None else out


class StaticGraphEmbedding(nn.Module):
    def __init__(self, n_tokens, emb_size):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(n_tokens, emb_size))
        nn.init.uniform_(self.emb, -1 / math.sqrt(emb_size), 1 / math.sqrt(emb_size))

    def forward(self, token_index=None):
        return self.emb if token_index is None else self.emb[token_index]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * divisor), torch.cos(position * divisor)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)])


class TSLNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, unbiased=False, keepdim=True)
        return (x - mean) / (std + self.eps) * self.weight + self.bias


class PyGLinear(nn.Linear):
    """PyG Linear's default Kaiming/Glorot choices for the used nonlazy case."""
    def __init__(self, in_features, out_features, bias=True,
                 weight_initializer=None, bias_initializer=None):
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        if self.weight_initializer == "glorot":
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            if self.bias_initializer == "zeros":
                nn.init.zeros_(self.bias)
            else:
                nn.init.uniform_(self.bias, -1 / math.sqrt(self.in_features),
                                 1 / math.sqrt(self.in_features))

