#!/usr/bin/env python3
"""Checks target-aware physical allocation of a generic LM-head shard."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(args.tool),
        "--input", str(args.input),
        "--output", str(args.output),
        "--pipeline", "ftlpu-stablehlo-to-tensor",
        "--target-config", str(args.target_config),
    ], check=True)
    text = args.output.read_text(encoding="utf-8")
    expected = [
        "ftlpu.tensor.matmul",
        "m = 128 : i64",
        "n = 4096 : i64",
        "k = 576 : i64",
        "instruction_count = 922 : i64",
    ]
    missing = [value for value in expected if value not in text]
    if missing:
        raise AssertionError(f"LM-head Tensor IR is missing: {missing}")


if __name__ == "__main__":
    main()
