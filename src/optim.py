"""Cross-entropy, AdamW, the LR schedule and gradient clipping (manual S4)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... vocab"], targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    """Mean ``-z_y + logsumexp(z)`` over every position (manual S4.1).

    ``torch.logsumexp`` applies the max shift internally, and the target logit is
    gathered directly, so ``log(softmax(logits))`` is never materialized.
    """
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            "cross_entropy expects one target per logit row: logits have leading "
            f"shape {tuple(logits.shape[:-1])} but targets have {tuple(targets.shape)}"
        )
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - target_logits).mean()


def _validate_hyperparameters(
    lr: float, betas: tuple[float, float], eps: float, weight_decay: float
) -> None:
    if lr < 0.0:
        raise ValueError(f"AdamW requires lr >= 0, got {lr}")
    if eps < 0.0:
        raise ValueError(f"AdamW requires eps >= 0, got {eps}")
    if weight_decay < 0.0:
        raise ValueError(f"AdamW requires weight_decay >= 0, got {weight_decay}")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"AdamW requires 0 <= beta1 < 1, got {beta1}")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"AdamW requires 0 <= beta2 < 1, got {beta2}")


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay (manual S4.2).

    Decay multiplies the parameter directly rather than entering the gradient, so
    it is never rescaled by the adaptive denominator.

    Bias correction uses each parameter's own update count, so a parameter that
    skips updates is corrected for how many it actually received.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # Defaults, then every group: a group override must not bypass the constraints.
        _validate_hyperparameters(lr, betas, eps, weight_decay)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            _validate_hyperparameters(
                group["lr"], group["betas"], group["eps"], group["weight_decay"]
            )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001, ANN201 - PyTorch signature
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # The scheduler rewrites group["lr"] every update, so re-check the live
        # values. Validating all groups first keeps a bad group from leaving an
        # earlier one half-updated.
        for group in self.param_groups:
            _validate_hyperparameters(
                group["lr"], group["betas"], group["eps"], group["weight_decay"]
            )

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    # Skipped completely: no state entry, no decay, no t++.
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                update_count = state["step"]
                first_moment = state["exp_avg"]
                second_moment = state["exp_avg_sq"]

                first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                first_unbiased = first_moment / (1.0 - beta1**update_count)
                second_unbiased = second_moment / (1.0 - beta2**update_count)

                parameter.mul_(1.0 - lr * weight_decay)
                parameter.addcdiv_(
                    first_unbiased, second_unbiased.sqrt().add_(eps), value=-lr
                )

        return loss


def get_lr_cosine_schedule(
    step: int,
    learning_rate_max: float,
    learning_rate_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """Linear warmup then cosine decay to a floor (manual S4.3)."""
    for name, value in (
        ("step", step),
        ("warmup_steps", warmup_steps),
        ("cosine_steps", cosine_steps),
    ):
        if not isinstance(value, int):
            raise TypeError(
                f"get_lr_cosine_schedule requires an integer {name}, got "
                f"{type(value).__name__}"
            )
    if step < 0:
        raise ValueError(f"get_lr_cosine_schedule requires step >= 0, got {step}")
    if warmup_steps < 0:
        raise ValueError(
            f"get_lr_cosine_schedule requires warmup_steps >= 0, got {warmup_steps}"
        )
    if warmup_steps >= cosine_steps:
        raise ValueError(
            "get_lr_cosine_schedule requires warmup_steps < cosine_steps, got "
            f"warmup_steps={warmup_steps} and cosine_steps={cosine_steps}"
        )
    if not 0.0 <= learning_rate_min <= learning_rate_max:
        raise ValueError(
            "get_lr_cosine_schedule requires 0 <= learning_rate_min <= "
            f"learning_rate_max, got min={learning_rate_min} and "
            f"max={learning_rate_max}"
        )

    if step < warmup_steps:
        # Unreachable when warmup_steps == 0, so no zero division.
        return float(step) / warmup_steps * learning_rate_max
    if step <= cosine_steps:
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return learning_rate_min + cosine * (learning_rate_max - learning_rate_min)
    return float(learning_rate_min)


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> float:
    """Scale all gradients by one factor when their joint L2 norm exceeds
    ``max_l2_norm`` (manual S4.3). Returns the norm before clipping.
    """
    if max_l2_norm <= 0.0:
        raise ValueError(
            f"gradient_clipping requires max_l2_norm > 0, got {max_l2_norm}"
        )

    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return 0.0

    with torch.no_grad():
        # One norm over every gradient jointly, not per-parameter norms.
        squared_total = torch.stack(
            [gradient.detach().square().sum() for gradient in gradients]
        ).sum()
        total_norm = float(squared_total.sqrt())
        if total_norm > max_l2_norm:
            scale = max_l2_norm / (total_norm + eps)
            for gradient in gradients:
                gradient.detach().mul_(scale)
    return total_norm
