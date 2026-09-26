"""Primitive building blocks (manual S3.3-S3.4).

Weights follow the PyTorch convention: a ``d_in -> d_out`` map is stored as
``[d_out, d_in]`` and applied as ``x @ W.T``.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    """Bias-free linear map ``y = x W^T`` (manual S3.3.2)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "Linear requires positive in_features and out_features, got "
                f"in_features={in_features}, out_features={out_features}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # N(0, 2 / (d_in + d_out)) truncated at +/- 3 sigma (manual S3.3.1).
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expected final dimension in_features={self.in_features}, "
                f"got {x.shape[-1]}"
            )
        return x @ self.weight.transpose(-1, -2)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"


class Embedding(nn.Module):
    """Learned token-embedding table (manual S3.3.2)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError(
                "Embedding requires positive num_embeddings and embedding_dim, got "
                f"num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # N(0, 1) truncated at +/- 3 (manual S3.3.1).
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        if not torch.is_tensor(token_ids) or token_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        ):
            raise TypeError(
                f"Embedding expects an integer tensor of token IDs, got {type(token_ids)}"
                + (f" with dtype {token_ids.dtype}" if torch.is_tensor(token_ids) else "")
            )
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class RMSNorm(nn.Module):
    """``y = gamma * x / sqrt(mean(x^2) + eps)`` over the final axis (manual S3.4.1).

    No centering, no bias.
    """

    def __init__(
        self,
        d_model: int,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"RMSNorm requires a positive d_model, got {d_model}")
        self.d_model = d_model
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"RMSNorm expected final dimension d_model={self.d_model}, got {x.shape[-1]}"
            )
        in_dtype = x.dtype
        # Squaring overflows in fp16, so upcast low precision only; fp64 keeps
        # its precision, and the result returns in the caller's dtype.
        if in_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        result = x * inverse_rms * self.weight.to(x.dtype)
        return result.to(in_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, norm_eps={self.norm_eps}"


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """SiLU / Swish: ``x * sigmoid(x)`` (manual S3.4.2)."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """``W_down(SiLU(W_gate x) * (W_up x))``, three bias-free projections (manual S3.4.2)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.down(silu(self.gate(x)) * self.up(x))


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """Numerically stable softmax along ``dim`` (manual S3.4.4).

    Subtracting the per-slice maximum leaves the result unchanged but caps every
    exponent at ``exp(0) = 1``, so large logits cannot overflow.
    """
    shifted = x - x.amax(dim=dim, keepdim=True)
    exponentiated = torch.exp(shifted)
    return exponentiated / exponentiated.sum(dim=dim, keepdim=True)
