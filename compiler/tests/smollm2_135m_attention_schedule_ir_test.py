#!/usr/bin/env python3
"""Checks generic attention lowering creates a cycle-bounded Schedule IR."""

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
    subprocess.run([
        str(args.tool), "--input", str(args.input), "--output", str(args.output),
        "--pipeline", "ftlpu-stablehlo-to-schedule",
    ], check=True)
    text = args.output.read_text(encoding="utf-8")
    required = (
        "ftlpu.schedule.attention",
        'name = "qkv"',
        'name = "rope"',
        'name = "qk"',
        'name = "softmax"',
        'name = "pv"',
        'name = "o_proj"',
        "work_waves =",
        "projection_work =",
        'projection = "query"',
        'projection = "key"',
        'projection = "value"',
        'phase = "qk"',
        'phase = "pv"',
        "query_head = 0 : i64",
        "kv_head = 0 : i64",
        "base_row = 7600 : i64",
        "query_weight_dequant",
        "output_weight_dequant",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"Attention Schedule IR is missing: {missing}")
    if text.count("ftlpu.schedule.mem_queue") != 43056:
        raise AssertionError("attention schedule did not emit the complete MEM queue program")
    if text.count("ftlpu.schedule.mxm_queue") != 4896:
        raise AssertionError("attention schedule did not emit projection and QK MXM commands")
    if text.count("ftlpu.schedule.vxm") != 44544:
        raise AssertionError("attention schedule did not emit dequant, RoPE, and cast VXM commands")
    if text.count('opcode = "iw"') != 2448:
        raise AssertionError("attention schedule did not emit all projection and QK IW commands")
    for column in range(4):
        if f"weight_column = {column} : i64" not in text:
            raise AssertionError(f"MXM IW weight column {column} is missing")


if __name__ == "__main__":
    main()
