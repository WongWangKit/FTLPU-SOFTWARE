#!/usr/bin/env python3
"""Checks direct StableHLO-to-binary emission against the Command IR path."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command_ir = args.output_dir / "reference.command.mlir"
    reference = args.output_dir / "reference.ftlpu"
    direct = args.output_dir / "direct.ftlpu"
    common = [
        "--mxm-execution", "block8",
        "--target-config", str(args.target_config),
    ]
    run([
        str(args.opt), "--input", str(args.input),
        "--output", str(command_ir),
        "--pipeline", "ftlpu-stablehlo-to-commands", *common,
    ])
    run([
        str(args.translate), "--input", str(command_ir),
        "--output", str(reference),
    ])
    run([
        str(args.compile), "--input", str(args.input),
        "--output", str(direct), "--input-stage", "stablehlo", *common,
    ])
    expected = reference.read_bytes()
    actual = direct.read_bytes()
    if actual != expected:
        mismatch = next((index for index, pair in enumerate(zip(actual, expected))
                         if pair[0] != pair[1]), min(len(actual), len(expected)))
        raise AssertionError(
            f"direct binary differs at byte {mismatch}: "
            f"direct={len(actual)} reference={len(expected)}")
    print(f"direct binary parity passed: {len(actual)} bytes")


if __name__ == "__main__":
    main()
