#!/usr/bin/env python3
"""Checks the first StableHLO -> FTLPU IR lowering passes."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_lower(tool, source, output, lower_to):
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--input",
            str(source),
            "--output",
            str(output),
            "--lower-to",
            lower_to,
            "--tile",
            "20",
        ],
        check=True,
    )
    return output.read_text(encoding="utf-8")


def assert_contains(text, fragments):
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError(f"missing expected fragments: {missing}")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tensor_ir = run_lower(
        args.tool, args.input, args.work_dir / "matmul_320.tensor.mlir", "tensor"
    )
    stream_ir = run_lower(
        args.tool, args.input, args.work_dir / "matmul_320.stream.mlir", "stream"
    )
    schedule_ir = run_lower(
        args.tool, args.input, args.work_dir / "matmul_320.schedule.mlir", "schedule"
    )

    assert_contains(
        tensor_ir,
        [
            "ftlpu.tensor.matmul",
            "m = 320",
            "n = 320",
            "k = 320",
            "lhs_dtype = \"i8\"",
            "acc_dtype = \"i32\"",
        ],
    )
    assert_contains(
        stream_ir,
        [
            "ftlpu.stream.matmul_grid",
            "tile_m = 20",
            "m_tiles = 16",
            "n_tiles = 16",
            "k_tiles = 16",
            "ftlpu.stream.matmul_tile @m0_n0_k0",
        ],
    )
    assert_contains(
        schedule_ir,
        [
            "ftlpu.schedule.program",
            "ftlpu.schedule.mem_read",
            "ftlpu.schedule.mxm_load",
            "ftlpu.schedule.mxm_compute",
            "ftlpu.schedule.mem_write",
            "accumulate false",
            "accumulate true",
        ],
    )
    print(f"generated tensor IR: {args.work_dir / 'matmul_320.tensor.mlir'}")
    print(f"generated stream IR: {args.work_dir / 'matmul_320.stream.mlir'}")
    print(f"generated schedule IR: {args.work_dir / 'matmul_320.schedule.mlir'}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
