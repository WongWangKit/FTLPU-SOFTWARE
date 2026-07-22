#!/usr/bin/env python3
"""Checks that expired tensor allocations are merged and reused."""

import argparse
import re
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
    result_words = [
        int(word)
        for word in re.findall(
            r"result_address = \{[^}]*word = (\d+) : i64", text
        )
    ]
    if result_words != [0, 0, 0]:
        raise AssertionError(
            f"expected result row storage to be reused at words [0, 0, 0], got {result_words}"
        )
    if "\n      lhs_address" not in text or "\n      result_address" not in text:
        raise AssertionError("Tensor op attributes were not printed on separate lines")


if __name__ == "__main__":
    main()
