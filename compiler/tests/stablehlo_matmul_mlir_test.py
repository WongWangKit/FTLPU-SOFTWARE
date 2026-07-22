#!/usr/bin/env python3
"""Verifies the real MLIR StableHLO-to-FTLPU kernel pass."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.tool), "--input", str(args.input), "--output", str(args.output)], check=True)
    text = args.output.read_text(encoding="utf-8")
    expected = [
        '"ftlpu.kernel.matmul"',
        'm = 320 : i64',
        'n = 320 : i64',
        'k = 320 : i64',
        'unit = "MXM"',
    ]
    missing = [item for item in expected if item not in text]
    if missing:
        raise AssertionError(f"missing MLIR attributes: {missing}")
    if "stablehlo.dot_general" in text:
        raise AssertionError("StableHLO dot_general remained after lowering")
    forbidden = ["mxm_count", "vector_layout"]
    present = [item for item in forbidden if item in text]
    if present:
        raise AssertionError(f"unexpected scheduling attributes in Kernel IR: {present}")


if __name__ == "__main__":
    main()
