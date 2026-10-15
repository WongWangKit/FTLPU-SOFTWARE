#!/usr/bin/env python3
"""Export SmolLM2-135M's forward pass to a StableHLO program.

The exported graph consumes:
  input_ids:      tensor<batch_size x sequence_length x i64>
  attention_mask: tensor<batch_size x sequence_length x i64>

and returns logits. Generation and the KV cache are deliberately kept outside
the graph; this makes the artifact a simple, statically-shaped compiler input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compiler" / "tools"))
from ftlpu_frontend import export_model  # noqa: E402
from ftlpu_frontend.config import DEFAULT_MODEL  # noqa: E402
from ftlpu_frontend.pipeline import require_empty_output  # noqa: E402,F401


def __getattr__(name):
    if name == "LogitsOnly":
        from ftlpu_frontend.hf import LogitsOnly
        return LogitsOnly
    raise AttributeError(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smollm2-135m-stablehlo"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="float32 is the safest choice for a new downstream backend",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not access Hugging Face; require the model to be cached",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.sequence_length < 1:
        raise SystemExit("--batch-size and --sequence-length must be positive")

    export_model({
        "model": args.model,
        "load_dtype": args.dtype,
        "load_options": {"local_files_only": args.local_files_only},
        "inputs": [
            {"name": "input_ids", "shape": [args.batch_size, args.sequence_length], "dtype": "int64", "fill": 0},
            {"name": "attention_mask", "shape": [args.batch_size, args.sequence_length], "dtype": "int64", "fill": 1},
        ],
    }, args.output)

    manifest = {
        "model": args.model,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "parameter_dtype": args.dtype,
        "inputs": {
            "input_ids": [args.batch_size, args.sequence_length],
            "attention_mask": [args.batch_size, args.sequence_length],
        },
        "output": "logits",
        "kv_cache": False,
        "entry_point": "forward",
    }
    (args.output / "export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    mlir = args.output / "functions" / "forward.mlir"
    bytecode = args.output / "functions" / "forward.bytecode"
    print("Export complete.")
    print(f"  StableHLO text:     {mlir.resolve()}")
    print(f"  StableHLO bytecode: {bytecode.resolve()}")
    print(f"  Manifest:           {(args.output / 'export.json').resolve()}")


if __name__ == "__main__":
    main()
