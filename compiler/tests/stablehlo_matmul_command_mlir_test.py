#!/usr/bin/env python3
"""Verifies complete StableHLO-to-Command lowering for the 320x320 matmul."""

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
            "ftlpu-stablehlo-to-commands",
        ],
        check=True,
    )

    text = args.output.read_text(encoding="utf-8")
    required = [
        "ftlpu.command.mxm",
        'opcode = "compute"',
        "repeat_count = 320 : i64",
        "accumulator_address = 0 : i64",
        "accumulator_row_stride = 1 : i64",
        'accumulator_destination = "stream"',
        'accumulator_output_format = "fp32"',
        "output_stream_base = 0 : i64",
        "ftlpu.command.mem",
        'opcode = "write"',
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"missing Command IR details: {missing}")

    forbidden = [
        "stablehlo.dot_general",
        "ftlpu.kernel.matmul",
        "ftlpu.tensor.matmul",
        "ftlpu.stream.matmul",
        "ftlpu.schedule.mem_",
        "ftlpu.schedule.mxm_",
    ]
    present = [item for item in forbidden if item in text]
    if present:
        raise AssertionError(f"operations remained after Command lowering: {present}")


if __name__ == "__main__":
    main()
