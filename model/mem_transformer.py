import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional

from .base import MemModel
# from model import Embeddings


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.view_as_real(
            torch.exp(1j * sinusoid_inp)).view(sinusoid_inp.shape[0], -1)
        return pos_emb


class RelativeMultiheadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=True)
        self.r_net = nn.Linear(d_model, d_head * n_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        # self.mu = nn.Parameter(torch.randn(n_head, d_head) * 0.02)
        # self.nu = nn.Parameter(torch.randn(n_head, d_head) * 0.02)

    def forward(self, x: Tensor, r: Tensor) -> Tensor:
        batch_size = x.size(0)
        ctx_size = r.size(0)

        head_k, head_v = self.kv_net(x).unfold(1, ctx_size, 1).chunk(2, 2)
        head_q = self.q_net(x[:, ctx_size-1:, :])
        q_size = head_q.size(1)
        head_r = self.r_net(r)

        mask = x[:, :ctx_size-1].eq(0).all(dim=-1)

        head_q = head_q.view(batch_size, q_size, self.n_head, self.d_head)
        head_k = head_k.reshape(
            batch_size, q_size, self.n_head, self.d_head, ctx_size)
        head_v = head_v.reshape(
            batch_size, q_size, self.n_head, self.d_head, ctx_size)
        head_r = head_r.view(ctx_size, self.n_head, self.d_head)

        attn_scores = head_q.unsqueeze(-2) @ (head_k + head_r.permute(1, 2, 0))
        attn_scores = attn_scores.squeeze(-2).transpose(1, 2) * self.scale

        # AC = (head_q + self.mu).unsqueeze(-2) @ head_k
        # # (batch_size, n_head, q_size, ctx_size)
        # AC = AC.squeeze(-2).transpose(1, 2)
        # # (batch_size, n_head, q_size, ctx_size)
        # BD = (head_q + self.nu).transpose(1, 2) @ head_r.permute(1, 2, 0)

        # attn_scores = (AC + BD).mul_(self.scale)

        if mask.any():
            mask = torch.cat(
                [mask, mask.new_zeros((batch_size, q_size))], dim=1)
            attn_mask = mask.unfold(1, ctx_size, 1)
            attn_scores = attn_scores.masked_fill(
                attn_mask.unsqueeze(1), -float('inf'))
        attn_weights = F.softmax(attn_scores, -1)

        attn_output = attn_weights.unsqueeze(
            -2) @ head_v.permute(0, 2, 1, 4, 3)
        attn_output = attn_output.squeeze(-2).transpose(
            1, 2).reshape(batch_size, q_size, -1)

        return self.dropout(self.o_net(attn_output))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_head, dropout=0.0) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.Attn = RelativeMultiheadAttention(
            n_head, d_model, d_head, dropout=dropout)

        self.FF = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, r: Tensor, y: Tensor) -> Tensor:
        tmp = self.Attn(self.attn_norm(x + y), r)
        x = x[:, -tmp.size(1):] + tmp
        return x + self.FF(x)


class WaveTransformer(MemModel):
    def __init__(self,
                 n_token: int,
                 d_model: int,
                 n_layer: int,
                 n_head: int,
                 aux_channels: int,
                 dropout: float = 0.1,
                 d_inner: int = 2048,
                 mem_len: int = 512) -> None:
        super().__init__(d_model, [mem_len - 1] * n_layer)

        self.n_token = n_token
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.d_inner = d_inner

        self.d_head = self.d_model // self.n_head
        self.aux_channels = aux_channels

        self.drop = nn.Dropout(self.dropout)
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.register_buffer(
            'pos', -torch.arange(0, mem_len, dtype=torch.float32).flip(0))

        self.layers = nn.ModuleList()
        for _ in range(self.n_layer):
            self.layers.append(
                DecoderLayer(self.d_model, self.d_inner,
                             self.n_head, self.d_head, self.dropout)
            )
        self.linear_proj = nn.Linear(self.d_model, self.n_token)
        self.condition = nn.Linear(aux_channels, self.d_model, bias=False)

        def weight_init(m: nn.Module):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif type(m) == nn.LayerNorm:
                nn.init.normal_(m.weight, 1.0, 0.02)

        self.apply(weight_init)

    def init_memories(self, batch=None, dtype=None, device=None) -> List[Tensor]:
        mems = super().init_memories(batch=batch, dtype=dtype, device=device)
        return [x.transpose(-1, -2) for x in mems] + [torch.zeros(batch, self.mem_sizes[-1], self.d_model, device=device, dtype=dtype)]

    def forward(self, x: Tensor, y: Tensor, memories: List[Tensor], **kwargs):
        pos = self.pos_emb(self.pos)
        x = self.drop(x).transpose(1, 2)
        cond = self.condition(y.transpose(1, 2))
        aug_cond = torch.cat([memories[-1], cond], dim=1)

        for i, layer in enumerate(self.layers):
            x = torch.cat([memories[i], x], dim=1)
            memories[i] = x[:, -pos.size(0)+1:].detach()
            x = layer(x, self.drop(pos), aug_cond)
        memories[-1] = aug_cond[:, -pos.size(0)+1:].detach()

        x = self.drop(x)
        pred = self.linear_proj(x).transpose(1, 2)
        return pred
