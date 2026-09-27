"""Figures for REPORT.md from one or more training logs (manual S7.2-S7.3).

Reads the ``metrics.csv`` written by ``src.train`` and writes PNGs under
``report_assets/``::

    uv run python -m src.plot --run baseline=runs/baseline/metrics.csv \
        --run "lr 6e-4=runs/high_lr/metrics.csv" --out_dir report_assets
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Jupyter hosts export MPLBACKEND=module://matplotlib_inline..., which a plain
# interpreter rejects at import time. Drop it before matplotlib reads it.
os.environ.pop("MPLBACKEND", None)

import matplotlib

# Non-interactive backend: these run headless on Kaggle and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def read_metrics(path: Path) -> dict[str, list[float]]:
    """Load one metrics.csv into parallel columns.

    ``val_loss`` is only written on evaluation steps, so it is returned as its
    own (shorter) series paired with its own steps.
    """
    steps: list[float] = []
    train_loss: list[float] = []
    learning_rate: list[float] = []
    grad_norm: list[float] = []
    val_steps: list[float] = []
    val_loss: list[float] = []

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = float(row["step"])
            steps.append(step)
            train_loss.append(float(row["train_loss"]))
            learning_rate.append(float(row["learning_rate"]))
            grad_norm.append(float(row["grad_norm"]))
            if row.get("val_loss"):
                val_steps.append(step)
                val_loss.append(float(row["val_loss"]))

    if not steps:
        raise ValueError(f"{path} contains no metric rows")
    return {
        "steps": steps,
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        "grad_norm": grad_norm,
        "val_steps": val_steps,
        "val_loss": val_loss,
    }


def plot_losses(runs: dict[str, dict], destination: Path, log_scale: bool) -> None:
    """Training and validation loss, overlaid across runs for ablation ladders."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, (label, series) in enumerate(runs.items()):
        color = f"C{index}"
        axis.plot(
            series["steps"],
            series["train_loss"],
            color=color,
            alpha=0.45,
            linewidth=1.0,
            label=f"{label} train",
        )
        if series["val_loss"]:
            axis.plot(
                series["val_steps"],
                series["val_loss"],
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=f"{label} val",
            )
    axis.set_xlabel("optimizer updates")
    axis.set_ylabel("cross-entropy (nats/token)")
    axis.set_title("Training and validation loss")
    if log_scale:
        axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_schedule_and_norms(runs: dict[str, dict], destination: Path) -> None:
    """Learning rate and pre-clipping gradient norm, the two stability diagnostics."""
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for index, (label, series) in enumerate(runs.items()):
        color = f"C{index}"
        top.plot(series["steps"], series["learning_rate"], color=color, label=label)
        bottom.plot(
            series["steps"], series["grad_norm"], color=color, alpha=0.8, label=label
        )
    top.set_ylabel("learning rate")
    top.set_title("Learning-rate schedule")
    top.grid(alpha=0.3)
    top.legend(fontsize=8)
    bottom.set_xlabel("optimizer updates")
    bottom.set_ylabel("grad norm (pre-clip)")
    bottom.set_title("Gradient norm before clipping")
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def parse_run(specification: str) -> tuple[str, Path]:
    """Accept ``label=path`` or a bare ``path`` (labelled by its parent directory)."""
    if "=" in specification:
        label, _, raw = specification.partition("=")
        return label.strip(), Path(raw.strip())
    path = Path(specification)
    return path.parent.name or path.stem, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot PA1 training logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="[LABEL=]METRICS_CSV",
        help="repeat to overlay several runs on one figure",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("report_assets"))
    parser.add_argument(
        "--prefix", type=str, default="", help="prepended to each output filename"
    )
    parser.add_argument(
        "--log_scale", action="store_true", help="log-scale the loss axis"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict] = {}
    for specification in args.run:
        label, path = parse_run(specification)
        runs[label] = read_metrics(path)
        series = runs[label]
        best = min(series["val_loss"]) if series["val_loss"] else float("nan")
        print(
            f"{label:24s} {len(series['steps']):>5} rows, "
            f"final step {int(series['steps'][-1]):>6}, "
            f"final train {series['train_loss'][-1]:.4f}, best val {best:.4f}"
        )

    losses = args.out_dir / f"{args.prefix}loss.png"
    diagnostics = args.out_dir / f"{args.prefix}schedule_and_grad_norm.png"
    plot_losses(runs, losses, args.log_scale)
    plot_schedule_and_norms(runs, diagnostics)
    print(f"\nwrote {losses}\nwrote {diagnostics}")


if __name__ == "__main__":
    main()
