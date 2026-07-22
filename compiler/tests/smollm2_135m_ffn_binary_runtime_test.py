#!/usr/bin/env python3
"""Compiles the SmolLM2-135M FFN and validates its CModel result."""

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
    commands = args.output_dir / "ffn.command.mlir"
    binary = args.output_dir / "ffn.ftlpu"
    subprocess.run([
        str(args.opt), "--input", str(args.input), "--output", str(commands),
        "--pipeline", "ftlpu-stablehlo-to-commands",
    ], check=True)
    subprocess.run([
        str(args.translate), "--input", str(commands), "--output", str(binary),
    ], check=True)
    subprocess.run([str(args.runtime_test), str(binary)], check=True)


if __name__ == "__main__":
    main()
