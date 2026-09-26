"""The pre-norm Transformer block and the full language model (manual S3.4-S3.5)."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from src.attention import CausalGroupedQuerySelfAttention
from src.layers import Embedding, Linear, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm block: normed attention and normed SwiGLU, each in its own residual.

    The two RMSNorms are distinct modules with independent parameters.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.attention = CausalGroupedQuerySelfAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            context_length=context_length,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch sequence d_model"],
        token_positions: Int[Tensor, "... sequence"] | None = None,
    ) -> Float[Tensor, "batch sequence d_model"]:
        # Positions reach attention only; the feed-forward sublayer is position-wise.
        x = x + self.attention(self.attention_norm(x), token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer LM (manual S3.5).

    Embedding -> blocks -> final RMSNorm -> untied bias-free LM head. Returns raw
    logits: no softmax inside the model, no absolute position embeddings, no dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                context_length=context_length,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        # Untied: its own [vocab_size, d_model] matrix, not the embedding table.
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: Int[Tensor, "batch sequence"],
        token_positions: Int[Tensor, "... sequence"] | None = None,
    ) -> Float[Tensor, "batch sequence vocab"]:
        sequence = token_ids.shape[-1]
        if sequence < 1:
            raise ValueError("TransformerLM requires a non-empty sequence, got length 0")
        if sequence > self.context_length:
            raise ValueError(
                f"TransformerLM supports sequences of at most context_length="
                f"{self.context_length}, got {sequence}"
            )

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        """Total trainable parameter values (19,272,192 for the fixed config)."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
