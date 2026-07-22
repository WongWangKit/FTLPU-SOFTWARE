#!/usr/bin/env python3
"""Checks generic attention lowering assigns all CModel-visible MEM regions."""

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
        "--pipeline", "ftlpu-stablehlo-to-tensor",
    ], check=True)
    text = args.output.read_text(encoding="utf-8")
    required = (
        "ftlpu.tensor.attention", "query_weight =", "key_weight =", "value_weight =",
        "output_weight =", "query =", "key =", "value =", "rope =", "score =",
        "exp =", "probability =", "context =", "result =", "slices = [28, 29, 30, 31]",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"Tensor attention memory plan is missing: {missing}")


if __name__ == "__main__":
    main()
