#!/usr/bin/env python3
"""Checks the C++ pass pipeline for StableHLO matmul lowering."""

import argparse
import subprocess
from pathlib import Path


def run_pipeline(tool, source, output, pipeline):
    subprocess.run(
        [
            str(tool),
            "--input",
            str(source),
            "--output",
            str(output),
            "--pipeline",
            pipeline,
        ],
        check=True,
    )
    return output.read_text(encoding="utf-8")


def assert_contains(text, fragments):
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError(f"missing expected fragments: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tensor_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "matmul_320.tensor.mlir",
        "stablehlo-to-tensor",
    )
    stream_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "matmul_320.stream.mlir",
        "stablehlo-to-tensor,tensor-to-stream",
    )
    schedule_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "matmul_320.schedule.mlir",
        "stablehlo-to-tensor,tensor-to-stream,stream-to-schedule",
    )

    assert_contains(
        tensor_ir,
        ["ftlpu.tensor.matmul", "m = 320", "n = 320", "k = 320", "acc_dtype = \"i32\""],
    )
    assert_contains(
        stream_ir,
        ["ftlpu.stream.matmul_grid", "m_tiles = 16", "n_tiles = 16", "k_tiles = 16"],
    )
    assert_contains(
        schedule_ir,
        [
            "ftlpu.schedule.program",
            "ftlpu.schedule.mem_read",
            "ftlpu.schedule.mxm_load",
            "ftlpu.schedule.mxm_compute",
            "ftlpu.schedule.mem_write",
        ],
    )
    print(f"generated tensor IR: {args.work_dir / 'matmul_320.tensor.mlir'}")
    print(f"generated stream IR: {args.work_dir / 'matmul_320.stream.mlir'}")
    print(f"generated schedule IR: {args.work_dir / 'matmul_320.schedule.mlir'}")


if __name__ == "__main__":
    main()
