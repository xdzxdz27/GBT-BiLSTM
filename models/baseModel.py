import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Baseline(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

class BiLSTM_Baseline(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

class GRU_Baseline(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  

    def forward(self, x):
        out, _ = self.gru(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)  
        return x

class Transformer_Baseline(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_heads=2, num_layers=1, use_mlp=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_encoder(x)  
        out = self.transformer_encoder(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

class DLinear_Baseline(nn.Module):
    def __init__(self, input_dim=5, seq_len=31, channel_fuse=True):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.individual = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(input_dim)])
        self.channel_fuse = channel_fuse
        if channel_fuse:
            self.fuse = nn.Linear(input_dim, 1)
        else:
            self.register_parameter('dummy', None)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.seq_len and C == self.input_dim, "DLinear: seq_len or input_dim mismatch"
        outs = []
        for c in range(C):
            outs.append(self.individual[c](x[:, :, c]))
        H = torch.cat(outs, dim=1)
        if self.channel_fuse:
            return self.fuse(H)
        else:
            return H[:, :1]
class TiDEResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, p=0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y + self.shortcut(x)


class TiDE_Baseline(nn.Module):
    def __init__(self, input_dim=5, seq_len=31, hidden=256, p=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.pred_len = 1

        self.temporal_width = input_dim
        encoder_input_dim = seq_len * (input_dim + self.temporal_width)
        temporal_hidden = max(hidden // 2, 64)

        self.feature_proj = nn.Sequential(
            TiDEResidualBlock(input_dim, hidden, self.temporal_width, p=p),
            TiDEResidualBlock(self.temporal_width, hidden, self.temporal_width, p=p),
        )

        self.encoder = nn.Sequential(
            TiDEResidualBlock(encoder_input_dim, hidden, hidden, p=p),
            TiDEResidualBlock(hidden, hidden, hidden, p=p),
        )

        self.decoder = nn.Sequential(
            TiDEResidualBlock(hidden, hidden, hidden, p=p),
            TiDEResidualBlock(hidden, hidden, self.pred_len * hidden, p=p),
        )

        self.temporal_decoder = nn.Sequential(
            TiDEResidualBlock(hidden + self.temporal_width, hidden, temporal_hidden, p=p),
            TiDEResidualBlock(temporal_hidden, hidden, 1, p=p),
        )

        self.global_residual = nn.Linear(seq_len * input_dim, self.pred_len)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.seq_len and C == self.input_dim, "TiDE: seq_len or input_dim mismatch"

        x_raw = x.reshape(B, -1)

        cov_proj = self.feature_proj(x.reshape(B * L, C)).reshape(B, L, self.temporal_width)
        encoder_input = torch.cat([x, cov_proj], dim=-1).reshape(B, -1)

        encoded = self.encoder(encoder_input)
        decoded = self.decoder(encoded).reshape(B, self.pred_len, -1)

        future_cov = torch.zeros(
            B, self.pred_len, self.temporal_width,
            device=x.device, dtype=x.dtype
        )
        temporal_input = torch.cat([decoded, future_cov], dim=-1).reshape(B * self.pred_len, -1)
        y_non_linear = self.temporal_decoder(temporal_input).reshape(B, self.pred_len)

        y_linear = self.global_residual(x_raw)
        return y_non_linear + y_linear


class MixerBlock(nn.Module):
    def __init__(self, L, C, r_token=2.0, r_channel=2.0, p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.token_mlp = nn.Sequential(
            nn.Linear(L, int(L * r_token)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(L * r_token), L),
            nn.Dropout(p),
        )
        self.ln2 = nn.LayerNorm(C)
        self.channel_mlp = nn.Sequential(
            nn.Linear(C, int(C * r_channel)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(C * r_channel), C),
            nn.Dropout(p),
        )

    def forward(self, x):
        y = self.ln1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y
        z = self.ln2(x)
        z = self.channel_mlp(z)
        return x + z


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._cached_mean = None
        self._cached_std = None

    def forward(self, x, mode="norm"):
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False)
            std = std.clamp_min(self.eps)
            self._cached_mean = mean
            self._cached_std = std
            x = (x - mean) / std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == "denorm":
            mean, std = self._cached_mean, self._cached_std
            if mean is None or std is None:
                return x
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            return x * std + mean
        else:
            raise ValueError("RevIN mode must be 'norm' or 'denorm'")

class MixerBlock(nn.Module):
    def __init__(self, L, d_model, r_token=2.0, r_channel=2.0, p=0.1):
        super().__init__()
        self.L = L
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(L, int(L * r_token)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(L * r_token), L),
            nn.Dropout(p),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * r_channel)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(d_model * r_channel), d_model),
            nn.Dropout(p),
        )

    def forward(self, x):
        y = self.ln1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        z = self.ln2(x)
        z = self.channel_mlp(z)
        return x + z


class TSMixer_Baseline(nn.Module):
    def __init__(
        self,
        input_dim=5,
        seq_len=31,
        depth=3,
        d_model=64,
        r_token=2.0,
        r_channel=2.0,
        p=0.1,
        enable_revin=False,
        revin_affine=True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.enable_revin = enable_revin

        self.revin = RevIN(input_dim, affine=revin_affine) if enable_revin else None

        self.in_proj = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList([
            MixerBlock(seq_len, d_model, r_token=r_token, r_channel=r_channel, p=p)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.seq_len and C == self.input_dim, "TSMixer: seq_len or input_dim mismatch"

        if self.enable_revin:
            x = self.revin(x, mode="norm")

        h = self.in_proj(x)

        for blk in self.blocks:
            h = blk(h)

        h = self.out_norm(h)
        h = h.mean(dim=1)
        y = self.proj(h)

        return y
    

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        self.mean = None
        self.stdev = None

    def forward(self, x, mode: str):
        if mode == 'norm':
            dim2reduce = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
            return x
        else:
            raise NotImplementedError

class PatchTST_Baseline(nn.Module):
    def __init__(self, input_dim=5, seq_len=31, patch_len=8, stride=4,
                 d_model=128, nhead=4, num_layers=3, p=0.1, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred_len

        self.revin = RevIN(input_dim, affine=True)

        base = seq_len - patch_len
        pad_len = (stride - (base % stride)) % stride
        self.pad_len = pad_len
        L_pad = seq_len + pad_len
        self.n_patches = (L_pad - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        self.dropout = nn.Dropout(p)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=p,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.n_patches * d_model, pred_len),
            nn.Dropout(p)
        )

        self.final_proj = nn.Linear(input_dim * pred_len, 1)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.seq_len and C == self.input_dim, "PatchTST: seq_len or input_dim mismatch"

        x = self.revin(x, 'norm')

        x = x.permute(0, 2, 1).reshape(B * C, L, 1)

        if self.pad_len > 0:
            x = F.pad(x, (0, 0, 0, self.pad_len), mode='replicate')

        x = x.squeeze(-1)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        x = self.patch_embedding(x)
        x = x + self.position_encoding
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        x = self.head(x)

        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        x = self.revin(x, 'denorm')

        x = x.reshape(B, -1)
        out = self.final_proj(x)
        return out

import torch
import torch.nn as nn

class iTransformer_Baseline(nn.Module):
    def __init__(self, input_dim=5, seq_len=31, d_model=128, nhead=4, num_layers=2, p=0.1, target_idx=0):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.target_idx = target_idx

        self.revin = RevIN(input_dim, affine=True)

        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.var_embedding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        nn.init.normal_(self.var_embedding, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=p, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.seq_len and C == self.input_dim, "iTransformer: seq_len or input_dim mismatch"

        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.enc_embedding(x)
        x = x + self.var_embedding[:, :C, :]
        x = self.dropout(x)

        x = self.encoder(x)
        h = x[:, self.target_idx, :]
        y = self.head(h)

        mean = self.revin.mean[:, :, self.target_idx]
        std = self.revin.stdev[:, :, self.target_idx]

        if getattr(self.revin, "affine", False):
            w = self.revin.affine_weight[self.target_idx].view(1, 1)
            b = self.revin.affine_bias[self.target_idx].view(1, 1)
            y = (y - b) / (w + self.revin.eps)

        y = y * std + mean
        return y
