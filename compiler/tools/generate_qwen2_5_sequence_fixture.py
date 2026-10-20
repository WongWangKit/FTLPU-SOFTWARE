#!/usr/bin/env python3
"""Generate an intermediate Qwen2.5 decoder-layer sequence fixture.

The checked-in seq32 and seq128 fixtures define which integer dimensions
represent sequence length.  Unchanged 128-wide head dimensions, f32/i32 types,
and model widths are left intact.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/qwen2_5_1_5b_decoder_layer"
SEQUENCE_DIMENSION = re.compile(r"(?<![A-Za-z0-9])32(?![0-9])|(?<=x)32(?![0-9])")


def transform(line: str, length: int) -> str:
    return SEQUENCE_DIMENSION.sub(str(length), line).replace("seq32", f"seq{length}")


def generate(length: int) -> str:
    if length < 32 or length > 128 or length % 32:
        raise ValueError("sequence length must be 32, 64, 96, or 128")
    short = (EXAMPLES / "decoder_layer_seq32.stablehlo.mlir").read_text(
        encoding="utf-8").splitlines(keepends=True)
    long = (EXAMPLES / "decoder_layer_seq128.stablehlo.mlir").read_text(
        encoding="utf-8").splitlines(keepends=True)
    if len(short) != len(long):
        raise ValueError("Qwen2.5 reference fixtures have different line counts")
    output = []
    for line32, line128 in zip(short, long):
        if line32 == line128:
            output.append(line32)
        elif transform(line32, 128) == line128:
            output.append(transform(line32, length))
        else:
            raise ValueError(f"unrecognized fixture difference: {line32.strip()}")
    return "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(args.seq_len), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
