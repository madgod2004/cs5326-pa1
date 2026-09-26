"""Train the Transformer LM on a flat token stream (manual S5.3).

The loop follows the canonical order from S5.3.2: train mode -> lr -> zero grads
-> n_acc microbatches -> backward accumulation -> clip -> optimizer update ->
validation -> logging -> checkpoint.

    uv run python -m src.train \
        --train_path data/tinystories/data/train.bin \
        --val_path data/tinystories/data/validation.bin \
        --out_dir runs/baseline --amp
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.data import get_batch, load_checkpoint, load_token_array, save_checkpoint
from src.model import TransformerLM
from src.optim import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping

CSV_FIELDS = (
    "step",
    "train_loss",
    "val_loss",
    "learning_rate",
    "grad_norm",
    "tokens",
    "elapsed_seconds",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a decoder-only Transformer LM (CS 5326 PA1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    architecture = parser.add_argument_group("model architecture")
    architecture.add_argument("--vocab_size", type=int, default=8192)
    architecture.add_argument("--context_length", type=int, default=256)
    architecture.add_argument("--d_model", type=int, default=512)
    architecture.add_argument("--num_layers", type=int, default=4)
    architecture.add_argument("--n_q_heads", type=int, default=16)
    architecture.add_argument("--n_kv_heads", type=int, default=4)
    architecture.add_argument("--d_ff", type=int, default=1344)
    architecture.add_argument("--rope_theta", type=float, default=10_000.0)
    architecture.add_argument("--norm_eps", type=float, default=1e-5)

    optimization = parser.add_argument_group("optimizer and schedule")
    optimization.add_argument("--learning_rate_max", type=float, default=3e-4)
    optimization.add_argument("--learning_rate_min", type=float, default=3e-5)
    optimization.add_argument("--warmup_steps", type=int, default=200)
    optimization.add_argument(
        "--cosine_steps",
        type=int,
        default=9999,
        help="fixed by the assignment at num_steps - 1 for the full run",
    )
    optimization.add_argument("--beta1", type=float, default=0.9)
    optimization.add_argument("--beta2", type=float, default=0.95)
    optimization.add_argument("--adam_eps", type=float, default=1e-8)
    optimization.add_argument("--weight_decay", type=float, default=0.1)
    optimization.add_argument("--max_grad_norm", type=float, default=1.0)

    schedule = parser.add_argument_group("training budget")
    schedule.add_argument("--batch_size", type=int, default=16, help="microbatch size B")
    schedule.add_argument("--sequence_length", type=int, default=256)
    schedule.add_argument("--gradient_accumulation_steps", type=int, default=16)
    schedule.add_argument("--num_steps", type=int, default=10_000, help="optimizer updates")

    streams = parser.add_argument_group("data and output")
    streams.add_argument("--train_path", type=Path, required=True)
    streams.add_argument("--val_path", type=Path, required=True)
    streams.add_argument("--out_dir", type=Path, required=True)
    streams.add_argument("--resume", type=Path, default=None)

    seeds = parser.add_argument_group("seeds")
    seeds.add_argument("--seed", type=int, default=0, help="model initialization")
    seeds.add_argument("--train_seed", type=int, default=1234)
    seeds.add_argument("--val_seed", type=int, default=5678)

    cadence = parser.add_argument_group("evaluation cadence")
    cadence.add_argument("--eval_interval", type=int, default=200)
    cadence.add_argument("--log_interval", type=int, default=50)
    cadence.add_argument("--checkpoint_interval", type=int, default=1000)
    cadence.add_argument("--num_validation_batches", type=int, default=20)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", type=str, default="auto")
    runtime.add_argument(
        "--amp",
        action="store_true",
        help="mixed precision: bf16 where supported, otherwise fp16 with a GradScaler",
    )
    runtime.add_argument(
        "--overfit_one_batch",
        action="store_true",
        help="reuse one fixed minibatch forever; its loss should collapse (S5.3.2 check)",
    )
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_precision(device: torch.device, amp: bool):
    """Return an autocast context factory and a matching GradScaler.

    bf16 needs no loss scaling; fp16 does. A disabled GradScaler is a transparent
    pass-through, so the training loop needs no branches.
    """
    use_amp = amp and device.type == "cuda"
    if not use_amp:
        return (lambda: contextlib.nullcontext()), torch.amp.GradScaler(device.type, enabled=False)
    use_bfloat16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    factory = lambda: torch.autocast(device_type="cuda", dtype=dtype)  # noqa: E731
    return factory, torch.amp.GradScaler("cuda", enabled=not use_bfloat16)


@torch.inference_mode()
def evaluate_validation(
    model: torch.nn.Module,
    tokens: np.ndarray,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    generator: torch.Generator,
    num_batches: int,
    autocast_factory,
) -> float:
    """Mean cross-entropy over ``num_batches`` validation batches.

    Saves and restores the previous train/eval mode, so a mid-loop call cannot
    leave the model in eval mode for the next training step.
    """
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(num_batches):
            inputs, targets = get_batch(
                tokens, batch_size, sequence_length, device, generator
            )
            with autocast_factory():
                total += float(cross_entropy(model(inputs), targets))
        return total / num_batches
    finally:
        model.train(was_training)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Seeds initialization only; batch sampling uses the two explicit generators.
    torch.manual_seed(args.seed)

    train_tokens = load_token_array(args.train_path)
    val_tokens = load_token_array(args.val_path)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate_max,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # CPU generators, so a run resumed on other hardware draws the same batches.
    train_generator = torch.Generator().manual_seed(args.train_seed)
    val_generator = torch.Generator().manual_seed(args.val_seed)

    next_step = 0
    if args.resume is not None:
        next_step = load_checkpoint(
            args.resume, model, optimizer, train_generator, val_generator
        )
        print(f"resumed from {args.resume} at step {next_step}")

    tokens_per_update = (
        args.batch_size * args.sequence_length * args.gradient_accumulation_steps
    )
    print(
        f"device={device} | parameters={model.num_parameters():,} | "
        f"tokens/update={tokens_per_update:,} | "
        f"total sampled tokens={tokens_per_update * args.num_steps:,}"
    )
    if args.overfit_one_batch:
        print("overfit_one_batch: training on a single fixed minibatch")

    autocast_factory, scaler = build_precision(device, args.amp)

    metrics_path = args.out_dir / "metrics.csv"
    is_new_log = not metrics_path.exists()
    metrics_file = metrics_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_file, fieldnames=CSV_FIELDS)
    if is_new_log:
        writer.writeheader()
        metrics_file.flush()

    checkpoint_path = args.out_dir / "checkpoint.pt"
    fixed_batch = None
    started = time.perf_counter()
    validation_loss: float | None = None

    try:
        for step in range(next_step, args.num_steps):
            model.train()

            learning_rate = get_lr_cosine_schedule(
                step,
                args.learning_rate_max,
                args.learning_rate_min,
                args.warmup_steps,
                args.cosine_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            # Once per optimizer update, not per microbatch.
            optimizer.zero_grad(set_to_none=True)

            train_loss = 0.0
            for _ in range(args.gradient_accumulation_steps):
                if args.overfit_one_batch:
                    if fixed_batch is None:
                        fixed_batch = get_batch(
                            train_tokens,
                            args.batch_size,
                            args.sequence_length,
                            device,
                            train_generator,
                        )
                    inputs, targets = fixed_batch
                else:
                    inputs, targets = get_batch(
                        train_tokens,
                        args.batch_size,
                        args.sequence_length,
                        device,
                        train_generator,
                    )

                with autocast_factory():
                    microbatch_loss = cross_entropy(model(inputs), targets)
                # Dividing by n_acc makes the accumulated gradient the gradient
                # of the mean microbatch loss; the logged value stays unscaled.
                scaler.scale(microbatch_loss / args.gradient_accumulation_steps).backward()
                train_loss += microbatch_loss.detach().item()
            train_loss /= args.gradient_accumulation_steps

            # Must unscale before measuring, or the norm is scale * true_norm.
            scaler.unscale_(optimizer)
            grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            completed_steps = step + 1
            final_step = completed_steps == args.num_steps
            should_validate = final_step or completed_steps % args.eval_interval == 0
            should_log = should_validate or completed_steps % args.log_interval == 0
            should_checkpoint = (
                final_step or completed_steps % args.checkpoint_interval == 0
            )

            if should_validate:
                validation_loss = evaluate_validation(
                    model,
                    val_tokens,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    device=device,
                    generator=val_generator,
                    num_batches=args.num_validation_batches,
                    autocast_factory=autocast_factory,
                )

            if should_log:
                elapsed = time.perf_counter() - started
                seen = completed_steps * tokens_per_update
                writer.writerow(
                    {
                        "step": completed_steps,
                        "train_loss": f"{train_loss:.6f}",
                        "val_loss": "" if not should_validate else f"{validation_loss:.6f}",
                        "learning_rate": f"{learning_rate:.8e}",
                        "grad_norm": f"{grad_norm:.6f}",
                        "tokens": seen,
                        "elapsed_seconds": f"{elapsed:.2f}",
                    }
                )
                metrics_file.flush()
                reported = (
                    f" val {validation_loss:.4f}" if should_validate else ""
                )
                print(
                    f"step {completed_steps:>6}/{args.num_steps} | "
                    f"train {train_loss:.4f}{reported} | "
                    f"lr {learning_rate:.3e} | grad {grad_norm:.3f} | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

            if should_checkpoint:
                # Write beside the target and rename: an interrupted save cannot
                # destroy the last good checkpoint.
                staging = checkpoint_path.with_suffix(".pt.tmp")
                save_checkpoint(
                    model,
                    optimizer,
                    completed_steps,
                    train_generator,
                    val_generator,
                    staging,
                )
                os.replace(staging, checkpoint_path)
    finally:
        metrics_file.close()

    total = time.perf_counter() - started
    print(
        f"done in {total:.0f}s | checkpoint at {checkpoint_path} | "
        f"metrics at {metrics_path}"
    )


if __name__ == "__main__":
    main()
