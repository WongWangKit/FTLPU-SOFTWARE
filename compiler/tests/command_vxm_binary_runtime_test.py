#!/usr/bin/env python3
"""Translates VXM Command IR and verifies runtime ICU queue loading."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(args.translate), "--input", str(args.input), "--output", str(args.output)
    ], check=True)
    subprocess.run([str(args.runtime_test), str(args.output)], check=True)


if __name__ == "__main__":
    main()
