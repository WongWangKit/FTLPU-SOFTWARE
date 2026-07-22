#!/usr/bin/env python3
"""Checks the canonical SmolLM2 attention entry lowers to generic Kernel IR."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(args.tool), "--input", str(args.input), "--output", str(args.output),
        "--pipeline", "ftlpu-stablehlo-to-kernel",
    ], check=True)
    text = args.output.read_text(encoding="utf-8")
    required = (
        "ftlpu.kernel.attention",
        "seq_len = 128 : i64",
        "hidden = 576 : i64",
        "query_heads = 9 : i64",
        "kv_heads = 3 : i64",
        "head_dim = 64 : i64",
        "causal = true",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"Kernel attention IR is missing: {missing}")


if __name__ == "__main__":
    main()
