import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 33

torch.manual_seed(SEED)


class PositionalEncodingLearned(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class GBFusion(nn.Module):
    def __init__(
        self,
        input_dim=4,
        lstm_hidden=64,
        d_model=64,
        nhead=4,
        ffn_mult=4,
        dropout=0.1,
        attn_heads=4,
        attn_temp=1.0,
        gate_tau=2.0,
        gate_lambda=0.3,
        use_radius_pe: bool = False,
        radius_in_dim: int = 1,
        global_encoder: str = "transformer",
    ):
        super().__init__()
        assert global_encoder in ["transformer", "mean_only"]

        self.global_encoder = global_encoder
        self.attn_heads = attn_heads
        self.attn_temp = attn_temp
        self.gate_tau = gate_tau
        self.gate_lambda = gate_lambda
        self.use_radius_pe = use_radius_pe
        self.radius_in_dim = radius_in_dim

        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.local_dim = lstm_hidden * 2

        self.proj_gb = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pos = PositionalEncodingLearned(d_model)

        if self.use_radius_pe:
            self.radius_mlp = nn.Sequential(
                nn.Linear(radius_in_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        D = d_model
        H = attn_heads
        assert D % H == 0, "d_model must be divisible by attn_heads"
        self.ap_query = nn.Parameter(torch.randn(H, D // H))
        self.ap_wq = nn.Linear(D, D, bias=False)
        self.ap_wk = nn.Linear(D, D, bias=False)
        self.ap_wv = nn.Linear(D, D, bias=False)
        self.ap_wo = nn.Linear(D, D, bias=False)

        self.align_for_feat = nn.Linear(d_model, self.local_dim)
        h = self.local_dim
        self.g_u = nn.Linear(h, h, bias=False)
        self.g_v = nn.Linear(h, h, bias=False)
        self.g_w = nn.Parameter(torch.randn(h))
        self.g_bet = nn.Parameter(torch.randn(1))
        self.g_out = nn.Linear(h, h, bias=True)

        self.post = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, 1),
        )

        self.head_student = nn.Sequential(
            nn.Linear(self.local_dim, self.local_dim // 2),
            nn.ReLU(),
            nn.Linear(self.local_dim // 2, 1),
        )

    def encode_global_seq(
        self,
        G_seq_global: torch.Tensor,
        r_seq: torch.Tensor = None,
    ) -> torch.Tensor:
        if G_seq_global.dim() == 2:
            G_seq_global = G_seq_global.unsqueeze(0)

        Z = self.proj_gb(G_seq_global)
        if self.global_encoder == "mean_only":
            return Z.mean(dim=1).squeeze(0)

        B, L, D = Z.shape

        if (not self.use_radius_pe) or (r_seq is None):
            Z = self.pos(Z)
        else:
            pe = self.pos.pe[:, :L, :]
            if r_seq.dim() == 1:
                r_seq = r_seq.unsqueeze(0).unsqueeze(-1)
            elif r_seq.dim() == 2:
                r_seq = r_seq.unsqueeze(-1)

            r_norm = torch.log(r_seq + 1e-6)
            r_emb = self.radius_mlp(r_norm)
            Z = Z + pe + r_emb

        Z = self.transformer(Z)

        H = self.attn_heads
        dh = D // H
        Q = self.ap_wq(Z).view(B, L, H, dh).transpose(1, 2)
        K = self.ap_wk(Z).view(B, L, H, dh).transpose(1, 2)
        V = self.ap_wv(Z).view(B, L, H, dh).transpose(1, 2)
        q = self.ap_query.view(1, H, 1, dh)

        attn_logits = (q * Q).sum(-1) / (dh ** 0.5)
        if self.attn_temp != 1.0:
            attn_logits = attn_logits / self.attn_temp
        attn = attn_logits.softmax(dim=-1)

        z_h = (attn.unsqueeze(-1) * V).sum(dim=2)
        z = z_h.transpose(1, 2).contiguous().view(B, D)
        z = self.ap_wo(z)
        return z.squeeze(0)

    def forward(
        self,
        X_raw: torch.Tensor,
        z_global_vec: torch.Tensor = None,
        gate_tau_override: float = None,
        gate_lambda_override: float = None,
        return_gate_aux: bool = False,
    ):
        H, _ = self.lstm(X_raw)
        z_local = H[:, -1, :]
        y_s = self.head_student(z_local)

        if z_global_vec is None:
            return {"y_s": y_s}

        B = X_raw.size(0)
        z_global = z_global_vec.unsqueeze(0).expand(B, -1)
        zg = self.align_for_feat(z_global)
        zl = z_local

        diff = torch.abs(zl - zg)
        cos = torch.clamp(
            (zl * zg).sum(-1) / (zl.norm(dim=-1) * zg.norm(dim=-1) + 1e-8),
            -1 + 1e-6,
            1 - 1e-6,
        )
        ang = 1.0 - cos

        w_pos = F.softplus(self.g_w)
        beta_pos = F.softplus(self.g_bet)
        logits = self.g_out(self.g_u(zl) + self.g_v(zg) - diff * w_pos)
        logits = logits - beta_pos * ang.unsqueeze(-1)

        tau = gate_tau_override if gate_tau_override is not None else self.gate_tau
        g = torch.sigmoid(logits / tau)

        lam = gate_lambda_override if gate_lambda_override is not None else self.gate_lambda
        z_f = (1 - lam + lam * g) * zl + lam * (1 - g) * zg
        y_gl = self.post(z_f)

        out = {"y_gl": y_gl, "y_s": y_s, "z_local": z_local, "z_global": z_global}
        return out
