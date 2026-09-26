"""Standardized validation and the final model export (manual S7.3-S7.4).

Two jobs, both reported in REPORT.md::

    uv run python -m src.evaluate --checkpoint runs/baseline/checkpoint.pt \
        --val_path data/tinystories/data/validation.bin --export final_model.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from src.data import get_batch, load_token_array
from src.generate import load_model
from src.optim import cross_entropy

# Fixed by manual S7.3 so every report number is comparable.
STANDARD_SEED = 42
STANDARD_NUM_BATCHES = 100
STANDARD_BATCH_SIZE = 16
STANDARD_SEQUENCE_LENGTH = 256

# manual S3.5 / S7.5
EXPECTED_PARAMETER_COUNT = 19_272_192


@torch.inference_mode()
def standardized_evaluation(
    model: torch.nn.Module,
    tokens: np.ndarray,
    *,
    seed: int = STANDARD_SEED,
    num_batches: int = STANDARD_NUM_BATCHES,
    batch_size: int = STANDARD_BATCH_SIZE,
    sequence_length: int = STANDARD_SEQUENCE_LENGTH,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Mean validation cross-entropy and perplexity (manual S7.3).

    The generator is created fresh here rather than passed in, so the reported
    number does not depend on how often validation happened during training.

    Runs in full precision: this is the number quoted in the report, so it
    should not vary with whichever autocast mode the training run used.

    Returns:
        ``(mean_cross_entropy_in_nats_per_token, perplexity)``.
    """
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(num_batches):
            inputs, targets = get_batch(
                tokens, batch_size, sequence_length, device, generator
            )
            total += float(cross_entropy(model(inputs), targets))
        mean_cross_entropy = total / num_batches
    finally:
        model.train(was_training)

    # Average the losses first, then exponentiate once. Averaging per-batch
    # perplexities would give a different (and larger) number.
    return mean_cross_entropy, math.exp(mean_cross_entropy)


def export_final_model(
    model: torch.nn.Module,
    destination: Path,
    expected_values: int | None = EXPECTED_PARAMETER_COUNT,
) -> int:
    """Write the model-only fp16 CPU state dict for submission (manual S7.4).

    Floating tensors become float16 and everything moves to CPU; non-floating
    tensors keep their dtype. Optimizer state, RNG state and hyperparameters are
    deliberately excluded -- this artifact is the model and nothing else.

    Returns:
        The number of tensor values written.
    """
    state = {
        name: tensor.detach().cpu().to(torch.float16)
        if tensor.is_floating_point()
        else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    total_values = sum(tensor.numel() for tensor in state.values())
    if expected_values is not None and total_values != expected_values:
        raise ValueError(
            f"Export holds {total_values:,} tensor values but the fixed "
            f"architecture requires exactly {expected_values:,}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return total_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardized validation and final-model export (CS 5326 PA1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val_path", type=Path, required=True)
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="also write the fp16 CPU state dict here, e.g. final_model.pt",
    )
    parser.add_argument(
        "--skip_parameter_check",
        action="store_true",
        help="allow exporting a model that is not the fixed 19,272,192-parameter one",
    )

    standard = parser.add_argument_group("evaluation protocol (fixed by S7.3)")
    standard.add_argument("--seed", type=int, default=STANDARD_SEED)
    standard.add_argument("--num_batches", type=int, default=STANDARD_NUM_BATCHES)
    standard.add_argument("--batch_size", type=int, default=STANDARD_BATCH_SIZE)
    standard.add_argument("--sequence_length", type=int, default=STANDARD_SEQUENCE_LENGTH)

    architecture = parser.add_argument_group("model architecture (must match the checkpoint)")
    architecture.add_argument("--vocab_size", type=int, default=8192)
    architecture.add_argument("--context_length", type=int, default=256)
    architecture.add_argument("--d_model", type=int, default=512)
    architecture.add_argument("--num_layers", type=int, default=4)
    architecture.add_argument("--n_q_heads", type=int, default=16)
    architecture.add_argument("--n_kv_heads", type=int, default=4)
    architecture.add_argument("--d_ff", type=int, default=1344)
    architecture.add_argument("--rope_theta", type=float, default=10_000.0)
    architecture.add_argument("--norm_eps", type=float, default=1e-5)

    parser.add_argument("--device", type=str, default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    model = load_model(
        args.checkpoint,
        device,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
    )
    tokens = load_token_array(args.val_path)

    print(
        f"device={device} | parameters={model.num_parameters():,} | "
        f"validation stream={len(tokens):,} tokens"
    )
    print(
        f"protocol: seed {args.seed}, {args.num_batches} batches of "
        f"{args.batch_size} x {args.sequence_length} "
        f"({args.num_batches * args.batch_size * args.sequence_length:,} tokens)"
    )

    mean_cross_entropy, perplexity = standardized_evaluation(
        model,
        tokens,
        seed=args.seed,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    print(f"\nvalidation cross-entropy = {mean_cross_entropy:.4f} nats/token")
    print(f"perplexity               = {perplexity:.4f}")

    if args.export is not None:
        written = export_final_model(
            model,
            args.export,
            expected_values=None if args.skip_parameter_check else EXPECTED_PARAMETER_COUNT,
        )
        size_mb = args.export.stat().st_size / 1024**2
        print(f"\nexported {written:,} values to {args.export} ({size_mb:.1f} MiB, fp16, CPU)")


if __name__ == "__main__":
    main()
