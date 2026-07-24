#!/usr/bin/env python3
"""Checks generic attention lowering creates a cycle-bounded Schedule IR."""

import argparse
import re
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
        'opcode = "max"',
        'opcode = "exp"',
        'opcode = "divide"',
        'opcode = "add"',
        'rhs_immediate = -1.000000e+09 : f32',
        'rhs_kind = "stream_f32"',
        'address = 8128 : i64',
        'address = 8158 : i64',
        'packed_stream = 48 : i64',
        'packed_stream = 49 : i64',
        'ftlpu.schedule.sxm',
        'opcode = "transpose"',
        'opcode = "permute"',
        'destination_streams = [32, 33, 34, 35',
        'weight_layout = "matrix_columns"',
        'opcode = "accumulator_read"',
        'accumulator_destination = "sram"',
        'accumulator_address = 3000 : i64',
        'output_stream = 8 : i64',
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"Attention Schedule IR is missing: {missing}")
    if 'opcode = "accumulate"' in text:
        raise AssertionError("attention schedule still uses legacy MEM accumulator commands")
    if text.count("ftlpu.schedule.mem_transfer") != 217662:
        raise AssertionError("attention schedule did not emit the complete physical MEM transfer program")
    if text.count("ftlpu.schedule.mxm_issue") != 18759:
        raise AssertionError("attention schedule did not emit all MXM compute and accumulator-read commands")
    if text.count('opcode = "accumulator_read"') != 9216:
        raise AssertionError("attention schedule did not read all MXM-resident accumulator rows")
    if text.count("ftlpu.schedule.vxm") != 122880:
        raise AssertionError("attention schedule did not emit projection, softmax, and context VXM commands")
    if text.count('rhs_immediate = -1.000000e+09 : f32') != 1728:
        raise AssertionError("attention schedule did not use immediate masks for all future blocks")
    mem_lines = [
        line for line in text.splitlines()
        if "ftlpu.schedule.mem_transfer" in line
    ]
    if (
        sum('address = 8128 : i64' in line for line in mem_lines) != 144
        or sum('address = 8158 : i64' in line for line in mem_lines) != 144
    ):
        raise AssertionError("attention schedule did not reuse the 31 diagonal causal-mask vectors")
    if text.count('opcode = "max"') != 4572:
        raise AssertionError("attention schedule did not emit recurrent softmax max commands")
    if text.count('opcode = "exp"') != 4608 or text.count('opcode = "divide"') != 4608:
        raise AssertionError("attention schedule did not emit complete softmax exp/divide commands")
    if text.count('opcode = "iw"') != 4032:
        raise AssertionError("attention schedule did not emit all projection, QK, and PV IW commands")
    if text.count("ftlpu.schedule.sxm") != 2172:
        raise AssertionError("attention schedule did not emit all probability and V transpose/permute waves")
    phase_matches = {
        name: (int(start), int(end))
        for end, name, start in re.findall(
            r'\{end = (\d+) : i64, name = "([^"]+)", start = (\d+) : i64\}',
            text,
        )
    }
    if phase_matches["qkv"][1] - phase_matches["qkv"][0] > 22238:
        raise AssertionError("Q/K/V projection lost its weight-prefetch pipeline")
    if phase_matches["o_proj"][1] - phase_matches["o_proj"][0] > 22428:
        raise AssertionError("O projection lost its weight-prefetch pipeline")
    for column in range(4):
        if f"weight_column = {column} : i64" not in text:
            raise AssertionError(f"MXM IW weight column {column} is missing")


if __name__ == "__main__":
    main()
