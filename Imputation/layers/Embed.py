import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LSTMTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(LSTMTokenEmbedding, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=c_in, hidden_size=d_model, batch_first=True)

        # Initialize the LSTM weights (optional)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        # x shape: (b, c_in, n)
        # Transpose the input to (b, n, c_in) to match LSTM input shape
        x = x.permute(0, 2, 1)  # (b, n, c_in)
        # Pass through LSTM
        x, _ = self.lstm(x)  # x shape: (b, n, d_model)

        return x


class SSMTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, layers=1, dropout=0.1):
        super(SSMTokenEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'in_proj': nn.Linear(1 if idx == 0 else d_model, d_model),
                'gate_proj': nn.Linear(1 if idx == 0 else d_model, d_model),
                'norm': nn.LayerNorm(d_model),
            })
            for idx in range(max(1, int(layers)))
        ])
        self.log_decay = nn.Parameter(torch.zeros(len(self.layers), d_model))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        # x shape: (b, c_in, n). Encode each flow's temporal window independently.
        bsz, steps, flows = x.shape
        series = x.permute(0, 2, 1).contiguous().view(bsz * flows, steps, 1)
        state = None
        current = series
        for layer_idx, layer in enumerate(self.layers):
            decay = torch.exp(-F.softplus(self.log_decay[layer_idx])).view(1, -1)
            hidden = current.new_zeros(current.shape[0], self.d_model)
            outputs = []
            for idx in range(current.shape[1]):
                token = current[:, idx, :]
                update = torch.tanh(layer['in_proj'](token))
                gate = torch.sigmoid(layer['gate_proj'](token))
                hidden = decay * hidden + (1.0 - decay) * update
                hidden = gate * hidden + (1.0 - gate) * update
                outputs.append(hidden)
            current = self.dropout(layer['norm'](torch.stack(outputs, dim=1)))
            state = current[:, -1, :]
        return state.view(bsz, flows, self.d_model)


class TemporalConvTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=5, dropout=0.1):
        super(TemporalConvTokenEmbedding, self).__init__()
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        hidden = max(16, d_model // 4)
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        bsz, steps, flows = x.shape
        series = x.permute(0, 2, 1).contiguous().view(bsz * flows, 1, steps)
        features = self.net(series).mean(dim=-1)
        return self.norm(self.proj(features)).view(bsz, flows, -1)


class PatchTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_len=10, stride=5, dropout=0.1, n_heads=4):
        super(PatchTokenEmbedding, self).__init__()
        self.patch_len = max(1, int(patch_len))
        self.stride = max(1, int(stride))
        self.patch_proj = nn.Linear(self.patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, int(n_heads)),
            dim_feedforward=d_model * 2,
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        bsz, steps, flows = x.shape
        series = x.permute(0, 2, 1).contiguous().view(bsz * flows, 1, steps)
        if steps < self.patch_len:
            series = F.pad(series, (0, self.patch_len - steps), mode='replicate')
        patches = series.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)
        tokens = self.patch_proj(patches)
        tokens = self.dropout(tokens + self.position_embedding(tokens).to(tokens.device, tokens.dtype))
        encoded = self.encoder(tokens)
        return self.norm(encoded.mean(dim=1)).view(bsz, flows, -1)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        c_in,
        d_model,
        encoder='lstm',
        dropout=0.1,
        layers=1,
        patch_len=10,
        patch_stride=5,
        kernel_size=5,
        n_heads=4,
    ):
        super(TokenEmbedding, self).__init__()
        encoder = str(encoder).lower()
        if encoder == 'lstm':
            self.encoder = LSTMTokenEmbedding(c_in, d_model)
        elif encoder in ('ssm', 'ssm_lite'):
            self.encoder = SSMTokenEmbedding(c_in, d_model, layers=layers, dropout=dropout)
        elif encoder in ('tcn', 'conv'):
            self.encoder = TemporalConvTokenEmbedding(c_in, d_model, kernel_size=kernel_size, dropout=dropout)
        elif encoder in ('patch', 'patch_transformer'):
            self.encoder = PatchTokenEmbedding(
                c_in,
                d_model,
                patch_len=patch_len,
                stride=patch_stride,
                dropout=dropout,
                n_heads=n_heads,
            )
        else:
            raise ValueError(f"Unsupported flow token encoder: {encoder}")

    def forward(self, x):
        return self.encoder(x)

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self,
        c_in,
        d_model,
        embed_type='fixed',
        freq='h',
        dropout=0.1,
        flow_encoder='lstm',
        flow_encoder_layers=1,
        flow_patch_len=10,
        flow_patch_stride=5,
        flow_kernel_size=5,
        flow_encoder_heads=4,
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(
            c_in=c_in,
            d_model=d_model,
            encoder=flow_encoder,
            dropout=dropout,
            layers=flow_encoder_layers,
            patch_len=flow_patch_len,
            patch_stride=flow_patch_stride,
            kernel_size=flow_kernel_size,
            n_heads=flow_encoder_heads,
        )
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                             freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #     d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
