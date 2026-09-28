"""Autoregressive decoding with temperature and top-p sampling (manual S6).

    uv run python -m src.generate --checkpoint runs/baseline/checkpoint.pt \
        --tokenizer data/tinystories/tokenizer/tokenizer.json \
        --prompt "Once upon a time" --temperature 0.8 --top_p 0.9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from jaxtyping import Int
from torch import Tensor

from src.layers import softmax
from src.model import TransformerLM

EOT_TOKEN = "<|endoftext|>"
_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def generate(
    model: torch.nn.Module,
    prompt_ids: Int[Tensor, " prompt"],
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Int[Tensor, " total"]:
    """Sample a completion and return the prompt followed by it (manual S6).

    The sequence may grow past ``context_length``; each step feeds only the most
    recent ``context_length`` tokens. There is no KV cache, so every step
    recomputes the whole cropped context.
    """
    if not torch.is_tensor(prompt_ids) or prompt_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            "generate expects an integer tensor of prompt token IDs, got "
            + (str(prompt_ids.dtype) if torch.is_tensor(prompt_ids) else type(prompt_ids).__name__)
        )
    if prompt_ids.ndim != 1:
        raise ValueError(
            f"generate expects a one-dimensional prompt, got shape {tuple(prompt_ids.shape)}"
        )
    if prompt_ids.numel() == 0:
        raise ValueError("generate requires a non-empty prompt")
    if max_new_tokens < 0:
        raise ValueError(f"generate requires max_new_tokens >= 0, got {max_new_tokens}")
    if context_length <= 0:
        raise ValueError(f"generate requires context_length > 0, got {context_length}")
    if temperature <= 0.0:
        raise ValueError(f"generate requires temperature > 0, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"generate requires 0 < top_p <= 1, got {top_p}")
    if not hasattr(model, "context_length"):
        raise AttributeError(
            "generate requires the model to expose a context_length attribute"
        )
    if context_length != model.context_length:
        raise ValueError(
            f"generate received context_length={context_length} but the model "
            f"supports {model.context_length}"
        )

    if max_new_tokens == 0:
        return prompt_ids

    device = next(model.parameters()).device
    # Plain ints while decoding, so the result can be built outside the
    # inference_mode block and the caller never receives an inference tensor.
    tokens = [int(value) for value in prompt_ids.tolist()]

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                window = tokens[-context_length:]
                inputs = torch.tensor([window], dtype=torch.long, device=device)
                logits = model(inputs)[0, -1, :]
                probabilities = softmax(logits / temperature, dim=-1)

                # `cumulative - sorted` is the mass strictly before each entry,
                # so comparing it to top_p keeps the shortest prefix reaching
                # top_p, crossing token included, with no index shifting.
                sorted_probabilities, sorted_indices = torch.sort(
                    probabilities, descending=True
                )
                cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                keep = (cumulative - sorted_probabilities) < top_p
                filtered = torch.where(
                    keep, sorted_probabilities, torch.zeros_like(sorted_probabilities)
                )
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)

                # Sample a rank, then map it back to the real vocabulary ID.
                # multinomial needs the distribution on the generator's device,
                # so a CPU generator still works when the model is on CUDA.
                if generator is not None:
                    filtered = filtered.to(generator.device)
                rank = int(torch.multinomial(filtered, num_samples=1, generator=generator))
                token = int(sorted_indices[rank])
                tokens.append(token)

                if eot_token_id is not None and token == eot_token_id:
                    break  # keep the EOT token in the output
    finally:
        model.train(was_training)

    return torch.tensor(tokens, dtype=torch.long, device=prompt_ids.device)


def load_model(
    checkpoint: Path,
    device: torch.device,
    **architecture: float | int,
) -> TransformerLM:
    """Build a model and load either a training checkpoint or a plain state dict."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    model = TransformerLM(**architecture).to(device)
    # Strict, so a dimension mismatch fails loudly. float16 exports cast up here.
    model.load_state_dict(state)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample text from a trained PA1 Transformer LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="tokenizer.json; omit only when using --prompt_ids",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", action="append", help="repeatable text prompt")
    source.add_argument(
        "--prompt_ids",
        type=str,
        default=None,
        help="comma-separated token IDs, for use without a tokenizer",
    )

    decoding = parser.add_argument_group("decoding")
    decoding.add_argument("--max_new_tokens", type=int, default=256)
    decoding.add_argument("--temperature", type=float, default=1.0)
    decoding.add_argument("--top_p", type=float, default=1.0)
    decoding.add_argument("--num_samples", type=int, default=1)
    decoding.add_argument("--seed", type=int, default=42)
    decoding.add_argument(
        "--no_eot_stop",
        action="store_true",
        help="keep decoding past <|endoftext|> instead of stopping",
    )

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
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also append each sample to this JSON Lines file",
    )
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

    tokenizer = None
    eot_token_id = None
    if args.tokenizer is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        eot_token_id = tokenizer.token_to_id(EOT_TOKEN)
    if args.no_eot_stop:
        eot_token_id = None

    if args.prompt_ids is not None:
        prompts = [[int(part) for part in args.prompt_ids.split(",")]]
    else:
        if tokenizer is None:
            raise SystemExit("--prompt needs --tokenizer; use --prompt_ids otherwise")
        prompts = [
            tokenizer.encode(text, add_special_tokens=False).ids for text in args.prompt
        ]

    generator = torch.Generator().manual_seed(args.seed)
    print(
        f"device={device} | temperature={args.temperature} | top_p={args.top_p} | "
        f"max_new_tokens={args.max_new_tokens}"
    )

    records = []
    for index, ids in enumerate(prompts):
        label = args.prompt[index] if args.prompt else args.prompt_ids
        for sample in range(args.num_samples):
            output = generate(
                model,
                torch.tensor(ids, dtype=torch.long, device=device),
                args.max_new_tokens,
                args.context_length,
                temperature=args.temperature,
                top_p=args.top_p,
                eot_token_id=eot_token_id,
                generator=generator,
            )
            generated = output.tolist()
            text = (
                tokenizer.decode(generated, skip_special_tokens=False)
                if tokenizer is not None
                else None
            )
            print(f"\n--- prompt {index} sample {sample} ({len(generated) - len(ids)} new tokens) ---")
            print(text if text is not None else generated)
            records.append(
                {
                    "prompt": label,
                    "prompt_ids": ids,
                    "sample": sample,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "token_ids": generated,
                    "text": text,
                }
            )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"\nappended {len(records)} samples to {args.out}")


if __name__ == "__main__":
    main()
