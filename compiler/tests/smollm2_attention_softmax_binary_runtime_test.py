#!/usr/bin/env python3
"""Validates compiler-generated Q/K/V, RoPE, QK, and softmax on CModel."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_ir = args.output_dir / "attention.command.mlir"
    binary = args.output_dir / "attention.ftlpu"
    subprocess.run([str(args.opt), "--input", str(args.input), "--output",
                    str(command_ir), "--pipeline", "ftlpu-stablehlo-to-commands"], check=True)
    subprocess.run([str(args.translate), "--input", str(command_ir),
                    "--output", str(binary)], check=True)
    subprocess.run([str(args.runtime_test), str(binary)], check=True)


if __name__ == "__main__":
    main()
