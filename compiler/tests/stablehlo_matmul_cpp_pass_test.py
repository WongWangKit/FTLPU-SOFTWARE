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
    kernel_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "matmul_320.kernel.mlir",
        "stablehlo-to-kernel",
    )
    tensor_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "matmul_320.tensor.mlir",
        "stablehlo-to-kernel,kernel-to-tensor",
    )
    stream_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "matmul_320.stream.mlir",
        "stablehlo-to-kernel,kernel-to-tensor,tensor-to-stream",
    )
    schedule_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "matmul_320.schedule.mlir",
        "stablehlo-to-kernel,kernel-to-tensor,tensor-to-stream,stream-to-schedule",
    )

    assert_contains(
        kernel_ir,
        ["ftlpu.kernel.mxm_matmul", "unit = \"MXM\"", "mxm_count = 2", "tile_m = 20"],
    )
    assert_contains(
        tensor_ir,
        [
            "ftlpu.tensor.mem_buffer @A0",
            "allocation = {addr = [device = 0, hemi = \"east\", slice = 0, bank = 0, word = 0, byte = 0]",
            "ftlpu.tensor.mem_buffer @B0",
            "ftlpu.tensor.tile_plan",
        ],
    )
    assert_contains(
        stream_ir,
        [
            "ftlpu.stream.matmul_grid",
            "south_to_north_tiles = 20",
            "ftlpu.stream.channel @matmul0_lhs",
            "stream_id = 16",
            "direction = \"E\"",
            "producer = \"MEM:A0\"",
            "consumer = \"MXM:*:activation\"",
            "produce_cycle =",
            "consume_cycle =",
            "latency =",
        ],
    )
    assert_contains(
        schedule_ir,
        [
            "ftlpu.schedule.program",
            "ftlpu.schedule.mem_read_weight @matmul0_read_weight",
            "ftlpu.schedule.mxm_load @matmul0_load",
            "ftlpu.schedule.mem_read_activation @matmul0_read_activation",
            "ftlpu.schedule.mxm_compute @matmul0_compute",
            "ftlpu.schedule.mem_write @matmul0_write",
            "weight_streams = [0..15]",
            "activation_streams = [16..31]",
            "south_to_north_tiles = 20",
        ],
    )
    if schedule_ir.count("ftlpu.schedule.") != 6:
        raise AssertionError("schedule IR should contain exactly five stage ops plus the program op")
    print(f"generated kernel IR: {args.work_dir / 'matmul_320.kernel.mlir'}")
    print(f"generated tensor IR: {args.work_dir / 'matmul_320.tensor.mlir'}")
    print(f"generated stream IR: {args.work_dir / 'matmul_320.stream.mlir'}")
    print(f"generated schedule IR: {args.work_dir / 'matmul_320.schedule.mlir'}")


if __name__ == "__main__":
    main()
