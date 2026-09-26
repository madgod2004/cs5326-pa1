"""Token stream, batch sampling and checkpoints (manual S5.1-S5.2)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, IO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor

# Raw little-endian unsigned 16-bit token IDs, no header.
TOKEN_DTYPE = np.dtype("<u2")


def load_token_array(path: str | os.PathLike) -> np.memmap:
    """Memory-map a flat token stream read-only (manual S5.1).

    Returns a memmap, not an array: the stream is far too large to hold as
    int64, so the OS pages in only the windows the sampler touches.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No token file at {path}")
    size_in_bytes = path.stat().st_size
    if size_in_bytes % TOKEN_DTYPE.itemsize != 0:
        raise ValueError(
            f"Token file {path} has {size_in_bytes} bytes, which is not a whole "
            f"number of {TOKEN_DTYPE.itemsize}-byte token IDs"
        )
    return np.memmap(path, mode="r", dtype=TOKEN_DTYPE)


def get_batch(
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    sequence_length: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Int[Tensor, "batch sequence"], Int[Tensor, "batch sequence"]]:
    """Sample next-token input/target windows (manual S5.1).

    Every offset ``o`` with ``0 <= o < len(dataset) - sequence_length`` is valid,
    giving ``x = u[o : o+n]`` and ``y = u[o+1 : o+n+1]``. ``generator`` is the
    sole source of randomness, so runs are reproducible without the global seed.
    """
    if batch_size <= 0:
        raise ValueError(f"get_batch requires batch_size > 0, got {batch_size}")
    if sequence_length <= 0:
        raise ValueError(
            f"get_batch requires sequence_length > 0, got {sequence_length}"
        )
    if len(dataset) < sequence_length + 1:
        raise ValueError(
            f"get_batch needs at least sequence_length + 1 = {sequence_length + 1} "
            f"tokens to form one window, but the stream holds {len(dataset)}"
        )

    # Exclusive bound, so `len - n` admits the final window without overrunning.
    starts = torch.randint(
        0, len(dataset) - sequence_length, (batch_size,), generator=generator
    )

    # Widen only the sampled slices; converting the whole stream would defeat the memmap.
    windows = np.stack(
        [
            dataset[start : start + sequence_length + 1].astype(np.int64)
            for start in starts.tolist()
        ]
    )
    batch = torch.from_numpy(windows)
    return batch[:, :-1].to(device), batch[:, 1:].to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize everything needed to resume the same trajectory (manual S5.2).

    ``next_step`` is the step that runs next, so a checkpoint taken after step
    ``s`` stores ``s + 1``. The generator states are included because without
    them a resumed run would replay batches it has already seen.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_step": int(next_step),
            "train_generator": train_generator.get_state(),
            "val_generator": val_generator.get_state(),
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
) -> int:
    """Restore a checkpoint in place and return its ``next_step``.

    Loading onto CPU first keeps generator states as the CPU byte tensors
    ``set_state`` requires; ``load_state_dict`` then copies onto the live devices.
    """
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_generator.set_state(checkpoint["train_generator"])
    val_generator.set_state(checkpoint["val_generator"])
    return int(checkpoint["next_step"])
