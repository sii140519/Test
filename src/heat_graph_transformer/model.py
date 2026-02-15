from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.edge_bias = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_heads),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D], positions: [B, N]
        bsz, n_nodes, _ = h.shape

        q = self.q_proj(h).view(bsz, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        rel = positions[:, :, None] - positions[:, None, :]
        rel = rel.abs().unsqueeze(-1)
        bias = self.edge_bias(rel).permute(0, 3, 1, 2)
        attn_score = attn_score + bias

        attn = torch.softmax(attn_score, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n_nodes, self.d_model)
        return self.out_proj(out)


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadGraphAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        hidden = d_model * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = h + self.dropout(self.attn(self.norm1(h), positions))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h


class GraphTransformerHeatModel(nn.Module):
    def __init__(
        self,
        in_dim: int = 7,
        d_model: int = 128,
        depth: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [
                GraphTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, node_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(node_features)
        for blk in self.blocks:
            h = blk(h, positions)
        h = self.norm(h)
        return self.head(h).squeeze(-1)
