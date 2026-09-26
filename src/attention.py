"""RoPE, scaled dot-product attention and grouped-query self-attention
(manual S3.4.3-S3.4.5).
"""

from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from src.layers import Linear, softmax

_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


class RotaryPositionalEmbedding(nn.Module):
    """Adjacent-pair RoPE (manual S3.4.3).

    Coordinate pairs ``(x_1, x_2), (x_3, x_4), ...`` are rotated by
    ``phi_{i,k} = i * Theta^(-(2k-2)/head_dim)``, which makes the query-key inner
    product depend only on the relative offset ``j - i``.

    This is the adjacent-pair convention, not the GPT-NeoX half-split one; the
    two must never be mixed. No learnable parameters.
    """

    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        context_length: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim so coordinates pair up, got {head_dim}"
            )
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.context_length = context_length

        pair_index = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device)
        inverse_frequencies = rope_theta ** (-pair_index / head_dim)
        positions = torch.arange(context_length, dtype=torch.float64, device=device)
        angles = positions[:, None] * inverse_frequencies[None, :]

        # float64 angles because i * omega grows with position; the finished
        # cos/sin are stored compactly in float32, one value per pair.
        self.register_buffer("cos_cache", angles.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_cache", angles.sin().to(torch.float32), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... sequence head_dim"],
        token_positions: Int[Tensor, "... sequence"],
    ) -> Float[Tensor, "... sequence head_dim"]:
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"RoPE expected a final dimension of head_dim={self.head_dim}, "
                f"got {x.shape[-1]}"
            )
        if not torch.is_tensor(token_positions) or token_positions.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                "RoPE expects integer token_positions, got "
                + (str(token_positions.dtype) if torch.is_tensor(token_positions) else type(token_positions).__name__)
            )
        if token_positions.numel() and (
            int(token_positions.min()) < 0 or int(token_positions.max()) >= self.context_length
        ):
            raise ValueError(
                "RoPE received token positions outside the cached range "
                f"[0, {self.context_length})"
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                f"RoPE expected one token position per sequence element: x has "
                f"sequence length {x.shape[-2]} but token_positions has "
                f"{token_positions.shape[-1]}"
            )

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # Position batch dims align with the LEADING dims of x, so pad singletons
        # in between. This is what lets [batch, sequence] positions broadcast
        # over the head axes of an x shaped [batch, heads, groups, seq, head_dim].
        x_batch_dims = x.ndim - 2
        position_batch_dims = token_positions.ndim - 1
        if x_batch_dims > position_batch_dims:
            broadcast_shape = (
                *token_positions.shape[:-1],
                *((1,) * (x_batch_dims - position_batch_dims)),
                token_positions.shape[-1],
                self.head_dim // 2,
            )
            cos = cos.reshape(broadcast_shape)
            sin = sin.reshape(broadcast_shape)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # `two` varies fastest, so index 0 is the even coordinate and 1 the odd.
        pairs = rearrange(x, "... sequence (pair two) -> ... sequence pair two", two=2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        )
        return rearrange(rotated, "... sequence pair two -> ... sequence (pair two)")

    def extra_repr(self) -> str:
        return (
            f"rope_theta={self.rope_theta}, head_dim={self.head_dim}, "
            f"context_length={self.context_length}"
        )


def scaled_dot_product_attention(
    queries: Float[Tensor, "... queries d_k"],
    keys: Float[Tensor, "... keys d_k"],
    values: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """``softmax(Q K^T / sqrt(d_k)) V`` with an optional mask (manual S3.4.4).

    ``True`` means the query may attend to that key. Leading dimensions
    broadcast, ``n_q`` need not equal ``n_kv``, and ``d_v`` need not equal ``d_k``.
    """
    d_k = queries.shape[-1]
    if keys.shape[-1] != d_k:
        raise ValueError(
            "Attention requires queries and keys to share a feature dimension, got "
            f"d_k={d_k} for queries and {keys.shape[-1]} for keys"
        )
    if values.shape[-2] != keys.shape[-2]:
        raise ValueError(
            "Attention requires keys and values to share a sequence length, got "
            f"{keys.shape[-2]} keys and {values.shape[-2]} values"
        )

    scores = einsum(queries, keys, "... query d_k, ... key d_k -> ... query key")
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        if not torch.is_tensor(mask) or mask.dtype != torch.bool:
            raise TypeError(
                "Attention requires a boolean mask, got "
                + (str(mask.dtype) if torch.is_tensor(mask) else type(mask).__name__)
            )
        # A fully masked query would make every score -inf and softmax NaN.
        if not bool(mask.any(dim=-1).all()):
            raise ValueError(
                "Attention mask leaves at least one query with no permitted key "
                "position, so the masked softmax is undefined"
            )
        scores = scores.masked_fill(~mask, float("-inf"))

    probabilities = softmax(scores, dim=-1)
    return einsum(probabilities, values, "... query key, ... key d_v -> ... query d_v")


class CausalGroupedQuerySelfAttention(nn.Module):
    """Causal, RoPE-enabled grouped-query self-attention (manual S3.4.5).

    ``n_q_heads`` query heads share ``n_kv_heads`` key/value heads in groups of
    ``g = n_q_heads // n_kv_heads``, with query head ``a = h * g + r`` belonging
    to key/value head ``h``. Covers MHA (``g == 1``), GQA and MQA
    (``n_kv_heads == 1``).
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        context_length: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_q_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_q_heads={n_q_heads}"
            )
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.queries_per_kv = n_q_heads // n_kv_heads  # g
        self.context_length = context_length

        self.q_proj = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(
            rope_theta=rope_theta,
            head_dim=self.head_dim,
            context_length=context_length,
            device=device,
        )

    def forward(
        self,
        x: Float[Tensor, "batch sequence d_model"],
        token_positions: Int[Tensor, "... sequence"] | None = None,
    ) -> Float[Tensor, "batch sequence d_model"]:
        sequence = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(sequence, device=x.device)

        # Nesting the head axis as (head group feature), feature fastest,
        # reproduces a = h * g + r exactly. The merge below must match.
        queries = rearrange(
            self.q_proj(x),
            "batch sequence (head group feature) -> batch head group sequence feature",
            head=self.n_kv_heads,
            group=self.queries_per_kv,
        )
        keys = rearrange(
            self.k_proj(x),
            "batch sequence (head feature) -> batch head sequence feature",
            head=self.n_kv_heads,
        )
        values = rearrange(
            self.v_proj(x),
            "batch sequence (head feature) -> batch head sequence feature",
            head=self.n_kv_heads,
        )

        # Queries and keys only; values carry no positional phase.
        queries = self.rope(queries, token_positions)
        keys = self.rope(keys, token_positions)

        # `group` appears only on the query side, so each key head is shared
        # across its g queries by the contraction, never by a materialized copy.
        scores = einsum(
            queries,
            keys,
            "batch head group query feature, batch head key feature -> batch head group query key",
        ) / math.sqrt(self.head_dim)

        # Causal over sequence indices, not token positions: j <= i.
        indices = torch.arange(sequence, device=x.device)
        causal = indices[:, None] >= indices[None, :]
        scores = scores.masked_fill(~causal, float("-inf"))

        probabilities = softmax(scores, dim=-1)
        attended = einsum(
            probabilities,
            values,
            "batch head group query key, batch head key feature -> batch head group query feature",
        )

        merged = rearrange(
            attended,
            "batch head group sequence feature -> batch sequence (head group feature)",
        )
        return self.out_proj(merged)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_q_heads={self.n_q_heads}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}"
        )
