#!/usr/bin/env python3
"""Checks that the Schedule verifier rejects physical resource overlap."""

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
    result = subprocess.run([
        str(args.tool), "--input", str(args.input), "--output", str(args.output),
        "--pipeline", "ftlpu-verify-schedule",
    ], capture_output=True, text=True)
    if result.returncode == 0:
        raise AssertionError("Schedule verifier accepted overlapping MEM transfers")
    if "overlaps at cycle 5" not in result.stderr:
        raise AssertionError(f"missing overlap diagnostic: {result.stderr}")


if __name__ == "__main__":
    main()
