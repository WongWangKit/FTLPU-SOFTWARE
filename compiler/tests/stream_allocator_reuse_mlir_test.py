#!/usr/bin/env python3
"""Checks direction-aware stream id allocation and lifetime reuse."""

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
            "ftlpu-stablehlo-to-stream",
        ],
        check=True,
    )
    text = args.output.read_text(encoding="utf-8")
    bindings = re.findall(
        r"stream_base = (\d+) : i64,\s+stream_count = (\d+) : i64,\s+register_id = (\d+) : i64,\s+direction = \"(east|west)\"",
        text,
    )
    expected = [
        ("0", "16", "1", "east"),
        ("16", "4", "9", "east"),
        ("0", "4", "11", "west"),
    ] * 3
    if bindings != expected:
        raise AssertionError(f"unexpected allocated/reused stream bindings: {bindings}")


if __name__ == "__main__":
    main()
