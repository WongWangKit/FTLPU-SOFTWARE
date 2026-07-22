#!/usr/bin/env python3
"""Verifies StableHLO-to-Tensor IR lowering and physical MEM allocation."""

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
    subprocess.run(
        [
            str(args.tool),
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--pipeline",
            "ftlpu-stablehlo-to-tensor",
        ],
        check=True,
    )
    text = args.output.read_text(encoding="utf-8")
    expected = [
        "ftlpu.tensor.matmul",
        "lhs_bytes = 102400 : i64",
        "rhs_bytes = 102400 : i64",
        "result_bytes = 409600 : i64",
        'lhs_address = {bank = 0 : i64, byte = 0 : i64, device = 0 : i64, hemisphere = "east", slice = 32 : i64, word = 0 : i64}',
        'kind = "vector", slices = [32]',
        'kind = "mxm_weight_striped", slices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]',
        'kind = "int32_byte_planar", slices = [40, 41, 42, 43]',
        "instruction_count = 20 : i64",
        "instruction_count = 320 : i64",
        "address_stride = -16 : i64",
    ]
    missing = [item for item in expected if item not in text]
    if missing:
        raise AssertionError(f"missing Tensor IR allocation details: {missing}")
    forbidden = ["stablehlo.dot_general", '"ftlpu.kernel.matmul"']
    present = [item for item in forbidden if item in text]
    if present:
        raise AssertionError(f"operations remained after Tensor IR lowering: {present}")

    roundtrip = args.output.with_suffix(".roundtrip.mlir")
    subprocess.run(
        [
            str(args.tool),
            "--input",
            str(args.output),
            "--output",
            str(roundtrip),
            "--pipeline",
            "ftlpu-stablehlo-to-kernel",
        ],
        check=True,
    )
    if 'ftlpu.tensor.matmul' not in roundtrip.read_text(encoding="utf-8"):
        raise AssertionError("multiline Tensor IR did not survive parse/print roundtrip")


if __name__ == "__main__":
    main()
