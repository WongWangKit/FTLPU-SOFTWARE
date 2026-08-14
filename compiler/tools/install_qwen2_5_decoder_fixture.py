#!/usr/bin/env python3
"""Installs a generated Qwen2.5 decoder StableHLO fixture into the source tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    required = ("tensor<128x1536xbf16>", "tensor<1536x8960xi8>",
                "tensor<128x12x128xbf16>", "dense<1.000000e+06>")
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise ValueError(f"input is not the expected Qwen fixture: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output)


if __name__ == "__main__":
    main()
