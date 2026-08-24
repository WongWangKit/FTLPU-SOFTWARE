#!/usr/bin/env python3
"""Runs the compact Block8 FFN wave-boundary numeric regression."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--weight-bank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_ir = args.output_dir / "ffn.command.mlir"
    binary = args.output_dir / "ffn.ftlpu"
    subprocess.run([
        str(args.opt), "--input", str(args.input),
        "--output", str(command_ir),
        "--pipeline", "ftlpu-stablehlo-to-commands",
        "--target-config", str(args.target_config),
        "--weight-bank", str(args.weight_bank),
        "--mxm-execution", "block8",
        "--ffn-schedule", "tail",
    ], check=True)
    subprocess.run([
        str(args.translate), "--input", str(command_ir),
        "--output", str(binary),
    ], check=True)
    subprocess.run([str(args.runtime_test), str(binary)], check=True)


if __name__ == "__main__":
    main()
