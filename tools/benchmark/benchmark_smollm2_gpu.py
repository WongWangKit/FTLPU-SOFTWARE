#!/usr/bin/env python3
"""Benchmark SmolLM2 prefill and optional token generation on one CUDA GPU."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=0,
        help="also benchmark autoregressive generation when greater than zero",
    )
    parser.add_argument(
        "--prompt",
        default="Explain in one sentence why the sky is blue.",
        help="prompt used by the optional generation benchmark",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not download model files from Hugging Face",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of a summary",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda"):
        raise SystemExit("--device must select a CUDA device, for example cuda:0")
    if args.batch_size < 1 or args.sequence_length < 1:
        raise SystemExit("--batch-size and --sequence-length must be positive")
    if args.warmup_runs < 0 or args.runs < 1 or args.new_tokens < 0:
        raise SystemExit("--warmup-runs and --new-tokens must be non-negative; --runs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered)) - 1)
    return ordered[index]


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "runs": len(values),
    }


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def benchmark_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    warmup_runs: int,
    runs: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    synchronize(device)

    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            start = time.perf_counter()
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
    return summarize_ms(elapsed_ms)


def benchmark_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    new_tokens: int,
    device: torch.device,
    warmup_runs: int,
    runs: int,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    generation_args = {
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "max_new_tokens": new_tokens,
        "min_new_tokens": new_tokens,
    }

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model.generate(**encoded, **generation_args)
    synchronize(device)

    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            start = time.perf_counter()
            model.generate(**encoded, **generation_args)
            synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    summary = summarize_ms(elapsed_ms)
    summary["prompt_tokens"] = encoded.input_ids.shape[1]
    summary["new_tokens"] = new_tokens
    summary["tokens_per_second"] = new_tokens / (float(summary["mean"]) / 1000.0)
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_start

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (args.batch_size, args.sequence_length),
        dtype=torch.int64,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    torch.cuda.reset_peak_memory_stats(device)
    result: dict[str, Any] = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_load_seconds": model_load_seconds,
        "prefill_ms": benchmark_prefill(
            model,
            input_ids,
            attention_mask,
            device,
            args.warmup_runs,
            args.runs,
        ),
    }

    if args.new_tokens:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
        )
        result["generation_ms"] = benchmark_generation(
            model,
            tokenizer,
            args.prompt,
            args.new_tokens,
            device,
            args.warmup_runs,
            args.runs,
        )

    result["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    prefill = result["prefill_ms"]
    print(f"Model: {result['model']}")
    print(f"GPU: {result['gpu']}")
    print(
        f"Input: batch={args.batch_size}, seq_len={args.sequence_length}, "
        f"dtype={args.dtype}"
    )
    print(f"Model load: {model_load_seconds:.3f} s")
    print(
        "Prefill: "
        f"mean={prefill['mean']:.3f} ms, "
        f"median={prefill['median']:.3f} ms, "
        f"p95={prefill['p95']:.3f} ms, "
        f"min={prefill['min']:.3f} ms, "
        f"max={prefill['max']:.3f} ms "
        f"({prefill['runs']} runs)"
    )
    if "generation_ms" in result:
        generation = result["generation_ms"]
        print(
            "Generation: "
            f"mean={generation['mean']:.3f} ms for {args.new_tokens} tokens, "
            f"{generation['tokens_per_second']:.2f} tokens/s"
        )
    print(f"Peak PyTorch memory: {result['peak_memory_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
