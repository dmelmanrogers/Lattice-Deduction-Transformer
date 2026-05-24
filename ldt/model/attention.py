from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LearnedPositionEmbedding(nn.Module):
    """Learned position embedding for fixed lattice positions.

    For 2D benchmark domains this uses separate row and column embeddings, matching
    the paper's learned 2D positional encoding. A 1D fallback is available for
    synthetic tests or future non-grid domains.
    """

    def __init__(self, num_positions: int, d_model: int, position_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.num_positions = num_positions
        self.d_model = d_model
        self.position_shape = position_shape

        if len(position_shape) == 2:
            rows, cols = position_shape
            if rows * cols != num_positions:
                raise ValueError("2D position_shape product must equal num_positions")
            self.row_embedding = nn.Embedding(rows, d_model)
            self.col_embedding = nn.Embedding(cols, d_model)
            row_idx, col_idx = torch.meshgrid(
                torch.arange(rows),
                torch.arange(cols),
                indexing="ij",
            )
            self.register_buffer("row_indices", row_idx.reshape(-1), persistent=False)
            self.register_buffer("col_indices", col_idx.reshape(-1), persistent=False)
            self.index_embedding = None
        elif len(position_shape) == 0:
            self.row_embedding = None
            self.col_embedding = None
            self.index_embedding = nn.Embedding(num_positions, d_model)
            self.register_buffer(
                "position_indices",
                torch.arange(num_positions),
                persistent=False,
            )
        else:
            raise ValueError("position_shape must be empty or 2D")

    def forward(self) -> Tensor:
        if self.index_embedding is not None:
            return self.index_embedding(self.position_indices).unsqueeze(0)

        assert self.row_embedding is not None
        assert self.col_embedding is not None
        return (
            self.row_embedding(self.row_indices)
            + self.col_embedding(self.col_indices)
        ).unsqueeze(0)


class RotaryEmbedding2D(nn.Module):
    """2D RoPE applied to position tokens, leaving the CLS token unchanged."""

    def __init__(
        self,
        num_positions: int,
        position_shape: tuple[int, ...],
        head_dim: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if len(position_shape) != 2:
            raise ValueError("2D RoPE requires a 2D position_shape")
        if head_dim % 4 != 0:
            raise ValueError("2D RoPE requires head_dim divisible by 4")

        rows, cols = position_shape
        if rows * cols != num_positions:
            raise ValueError("2D position_shape product must equal num_positions")

        axis_dim = head_dim // 2
        inv_freq = base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        row_idx, col_idx = torch.meshgrid(
            torch.arange(rows, dtype=torch.float32),
            torch.arange(cols, dtype=torch.float32),
            indexing="ij",
        )
        self.num_positions = num_positions
        self.axis_dim = axis_dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("row_indices", row_idx.reshape(-1), persistent=False)
        self.register_buffer("col_indices", col_idx.reshape(-1), persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        if query.shape != key.shape:
            raise ValueError("query and key must have identical shapes")
        if query.ndim != 4:
            raise ValueError("query and key must have shape (batch, heads, tokens, head_dim)")
        if query.shape[-2] != self.num_positions + 1:
            raise ValueError("2D RoPE expects one CLS token plus num_positions tokens")

        return self._apply(query), self._apply(key)

    def _apply(self, tensor: Tensor) -> Tensor:
        cls_token = tensor[:, :, :1, :]
        position_tokens = tensor[:, :, 1:, :]
        row_part, col_part = position_tokens.split(self.axis_dim, dim=-1)
        row_part = self._apply_axis(row_part, self.row_indices)
        col_part = self._apply_axis(col_part, self.col_indices)
        rotated = torch.cat([row_part, col_part], dim=-1)
        return torch.cat([cls_token, rotated], dim=-2)

    def _apply_axis(self, tensor: Tensor, positions: Tensor) -> Tensor:
        dtype = tensor.dtype
        device = tensor.device
        angles = positions.to(device=device, dtype=dtype)[:, None] * self.inv_freq.to(
            device=device,
            dtype=dtype,
        )[None, :]
        cos = angles.cos().repeat_interleave(2, dim=-1)[None, None, :, :]
        sin = angles.sin().repeat_interleave(2, dim=-1)[None, None, :, :]
        return (tensor * cos) + (_rotate_interleaved(tensor) * sin)


def _rotate_interleaved(tensor: Tensor) -> Tensor:
    rotated = torch.empty_like(tensor)
    rotated[..., 0::2] = -tensor[..., 1::2]
    rotated[..., 1::2] = tensor[..., 0::2]
    return rotated


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        *,
        num_positions: int,
        position_shape: tuple[int, ...],
        use_rope: bool,
        rope_base: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.rope = (
            RotaryEmbedding2D(num_positions, position_shape, self.head_dim, rope_base)
            if use_rope
            else None
        )

    def forward(self, hidden: Tensor) -> Tensor:
        batch, tokens, _ = hidden.shape
        qkv = self.qkv(hidden)
        qkv = qkv.view(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        if self.rope is not None:
            query, key = self.rope(query, key)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        context = attention @ value
        context = context.transpose(1, 2).contiguous().view(batch, tokens, self.d_model)
        return self.output_dropout(self.output(context))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_multiplier: float,
        dropout: float,
        *,
        num_positions: int,
        position_shape: tuple[int, ...],
        use_rope: bool,
        rope_base: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * ffn_multiplier)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            dropout,
            num_positions=num_positions,
            position_shape=position_shape,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.ffn(self.ffn_norm(hidden))
