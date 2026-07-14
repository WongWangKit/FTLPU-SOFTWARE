#!/usr/bin/env python3
"""Checks the C++ pass pipeline for StableHLO SwiGLU lowering."""

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
        args.work_dir / "swiglu.kernel.mlir",
        "stablehlo-to-kernel",
    )
    stream_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "swiglu.stream.mlir",
        "stablehlo-to-kernel,kernel-to-tensor,tensor-to-stream",
    )
    schedule_ir = run_pipeline(
        args.tool,
        args.input,
        args.work_dir / "swiglu.schedule.mlir",
        "stablehlo-to-kernel,kernel-to-tensor,tensor-to-stream,stream-to-schedule",
    )

    assert_contains(
        kernel_ir,
        [
            "ftlpu.kernel.mxm_matmul",
            "ftlpu.kernel.swiglu",
            "units = [\"MXM0\", \"MXM1\", \"VXM\"]",
            "rows = 160",
            "columns = 320",
        ],
    )
    assert_contains(
        stream_ir,
        [
            "ftlpu.stream.vxm_swiglu @swiglu0",
            "gate_stream = 32",
            "up_stream = 36",
            "output_stream = 31",
            "input_consume_cycle =",
            "producer = \"VXM:swiglu0:output\"",
            "consumer = \"MEM:S0\"",
            "ftlpu.stream.channel @swiglu0_out",
        ],
    )
    assert_contains(
        schedule_ir,
        [
            "ftlpu.schedule.mem_read_weight_gate_up @swiglu0_read_weight",
            "ftlpu.schedule.mxm_load_gate_up @swiglu0_load",
            "ftlpu.schedule.mem_read_activation @swiglu0_read_activation",
            "ftlpu.schedule.mxm_compute_gate_up @swiglu0_gate_up",
            "ftlpu.schedule.vxm_swiglu @swiglu0_vxm",
            "ftlpu.schedule.mem_write @swiglu0_write",
            "activation_stream = 16",
            "stages = 10",
            "output_scale = 0.0078125",
        ],
    )
    forbidden = [
        "ftlpu.schedule.mem_write @matmul0_write",
        "ftlpu.schedule.mem_write @matmul1_write",
    ]
    present = [fragment for fragment in forbidden if fragment in schedule_ir]
    if present:
        raise AssertionError(f"fused SwiGLU schedule should not write intermediate matmuls: {present}")
    print(f"generated kernel IR: {args.work_dir / 'swiglu.kernel.mlir'}")
    print(f"generated stream IR: {args.work_dir / 'swiglu.stream.mlir'}")
    print(f"generated schedule IR: {args.work_dir / 'swiglu.schedule.mlir'}")


if __name__ == "__main__":
    main()
