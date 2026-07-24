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
    source = args.input.read_text(encoding="utf-8")
    if "stablehlo.custom_call" in source:
        raise AssertionError("attention fixture must use a standard StableHLO graph")
    for operation in (
        "stablehlo.dot_general", "stablehlo.iota", "stablehlo.compare",
        "stablehlo.select", "stablehlo.reduce", "stablehlo.exponential",
        "stablehlo.power", "stablehlo.slice", "stablehlo.concatenate",
    ):
        if operation not in source:
            raise AssertionError(f"standard attention graph is missing {operation}")
    required = (
        "ftlpu.kernel.matmul",
        "ftlpu.kernel.rope",
        "ftlpu.kernel.gqa_broadcast",
        "ftlpu.kernel.transpose",
        "ftlpu.kernel.batch_matmul",
        "ftlpu.kernel.softmax",
        "kv_heads = 3 : i64",
        "head_dim = 64 : i64",
        "causal = true",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"Kernel attention IR is missing: {missing}")
    if "stablehlo." in text:
        raise AssertionError("matched attention graph was not fully consumed")
    if "ftlpu.kernel.attention" in text:
        raise AssertionError("attention must be decomposed in public Kernel IR")
    if text.count("ftlpu.kernel.matmul") != 4:
        raise AssertionError("attention must contain Q/K/V/O matmul operations")
    if text.count("ftlpu.kernel.rope") != 2:
        raise AssertionError("attention must contain separate Q and K RoPE operations")
    if text.count("ftlpu.kernel.batch_matmul") != 2:
        raise AssertionError("attention must contain QK and PV batched matmuls")


if __name__ == "__main__":
    main()
