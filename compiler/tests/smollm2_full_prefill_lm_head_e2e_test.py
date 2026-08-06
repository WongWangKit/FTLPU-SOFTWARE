#!/usr/bin/env python3
"""Runs seq128 full prefill and the quantized LPU LM head as one test."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--lm-head-binary", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for path in (
        args.runtime_test,
        args.package,
        args.lm_head_binary,
        args.weight,
        args.metadata,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    scale = float(metadata["quantization"]["hardware_bf16_scale"])
    result = subprocess.run(
        [
            str(args.runtime_test),
            str(args.package),
            str(args.lm_head_binary),
            str(args.weight),
            str(scale),
        ],
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    result.check_returncode()


if __name__ == "__main__":
    main()
