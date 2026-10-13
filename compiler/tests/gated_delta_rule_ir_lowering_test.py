#!/usr/bin/env python3
"""Checks the generic recurrent scan reaches hardware Command IR."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def lower(tool: Path, source: Path, output: Path, pipeline: str) -> str:
    subprocess.run(
        [str(tool), "--input", str(source), "--output", str(output),
         "--pipeline", pipeline],
        check=True,
    )
    return output.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tensor = lower(args.tool, args.input,
                   args.output_dir / "delta.tensor.mlir",
                   "ftlpu-stablehlo-to-tensor")
    for required in (
        "ftlpu.tensor.gated_delta_rule_task",
        'kind = "fp32_quad_planar"',
        'kind = "fp16_scalar_broadcast_planar"',
        'kind = "fp32_scalar_broadcast_planar"',
        'lowering = "direct_recurrent_scan"',
        "final_state_allocations",
    ):
        if required not in tensor:
            raise AssertionError(f"Tensor IR is missing {required!r}")

    stream = lower(args.tool, args.input,
                   args.output_dir / "delta.stream.mlir",
                   "ftlpu-stablehlo-to-stream")
    for required in (
        "ftlpu.stream.gated_delta_rule_task",
        "input_stream_bases",
        "output_stream_bases",
        'kind = "fp32_quad_planar"',
    ):
        if required not in stream:
            raise AssertionError(f"Stream IR is missing {required!r}")
    if "qwen" in stream.lower():
        raise AssertionError("model-specific name leaked into generic Stream IR")

    schedule = lower(args.tool, args.input,
                     args.output_dir / "delta.schedule.mlir",
                     "ftlpu-stablehlo-to-schedule")
    for required in (
        "ftlpu.schedule.mem_transfer",
        "ftlpu.schedule.vxm",
        'opcode = "multiply"',
        'opcode = "add"',
        'name = "gated_delta_rule"',
    ):
        if required not in schedule:
            raise AssertionError(f"Schedule IR is missing {required!r}")
    if "gated_delta_rule_task" in schedule:
        raise AssertionError("gated delta-rule task survived schedule lowering")

    command = lower(args.tool, args.input,
                    args.output_dir / "delta.command.mlir",
                    "ftlpu-stablehlo-to-commands")
    for required in (
        "ftlpu.command.mem_3d",
        "ftlpu.command.vxm_run_2d",
        'name = "gated_delta_rule"',
    ):
        if required not in command:
            raise AssertionError(f"Command IR is missing {required!r}")
    if "gated_delta_rule_task" in command:
        raise AssertionError("gated delta-rule task survived command lowering")
    if "qwen" in command.lower():
        raise AssertionError("model-specific name leaked into hardware IR")
    print("generic gated delta-rule lowered Kernel -> Tensor -> Stream -> "
          "Schedule -> Command")


if __name__ == "__main__":
    main()
