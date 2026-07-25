#!/usr/bin/env python3
"""Checks Tensor-to-Stream rejects an incomplete attention task graph."""

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
    result = subprocess.run(
        [
            str(args.tool),
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--pipeline",
            "ftlpu-stablehlo-to-stream",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        raise AssertionError("incomplete Tensor attention graph was accepted")
    diagnostic = result.stdout + result.stderr
    expected = "does not terminate a complete primitive attention task graph"
    if expected not in diagnostic:
        raise AssertionError(f"missing graph diagnostic:\n{diagnostic}")


if __name__ == "__main__":
    main()
