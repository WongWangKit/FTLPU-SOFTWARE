#!/usr/bin/env python3
"""Checks generic kernel-4 depthwise Conv1D through the real lowering path."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(tool: Path, source: Path, output: Path, pipeline: str) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(tool), "--input", str(source), "--output", str(output),
         "--pipeline", pipeline],
        check=True,
    )
    return output.read_text(encoding="utf-8")


def require(text: str, values: list[str], stage: str) -> None:
    missing = [value for value in values if value not in text]
    if missing:
        raise AssertionError(f"{stage} is missing {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tensor = run(args.tool, args.input, args.output_dir / "conv.tensor.mlir",
                 "ftlpu-stablehlo-to-tensor")
    require(tensor, ["ftlpu.tensor.depthwise_conv1d_task",
                     'lowering = "direct_closed_form"',
                     "scratch_allocations"], "Tensor IR")

    stream = run(args.tool, args.input, args.output_dir / "conv.stream.mlir",
                 "ftlpu-stablehlo-to-stream")
    require(stream, ["ftlpu.stream.depthwise_conv1d_task",
                     "input_stream_base", "weight_stream_base"], "Stream IR")

    schedule = run(args.tool, args.input, args.output_dir / "conv.schedule.mlir",
                   "ftlpu-stablehlo-to-schedule")
    require(schedule, ["ftlpu.schedule.mem_transfer",
                       "ftlpu.schedule.vxm",
                       'opcode = "multiply"',
                       'opcode = "add"',
                       'opcode = "reciprocal"',
                       'lhs_stream_source = "west"',
                       'name = "depthwise_conv1d"'], "Schedule IR")
    for line in schedule.splitlines():
        if "ftlpu.schedule.vxm" in line and 'input_hemisphere = "east"' not in line:
            raise AssertionError("increasing Conv1D VALU chain must propagate east")
    vxm_ops = [line for line in schedule.splitlines()
               if "ftlpu.schedule.vxm" in line]
    if len(vxm_ops) != 18 or any('opcode = "pass"' in line for line in vxm_ops):
        raise AssertionError("Conv1D must use one reduction tree and six-stage SiLU")
    mem_ops = [line for line in schedule.splitlines()
               if "ftlpu.schedule.mem_transfer" in line]
    if len(mem_ops) != 44:
        raise AssertionError("Conv1D introduced extra intermediate MEM transfers")
    require(schedule, ['opcode = "write_tap"'], "non-consuming replication")
    require(tensor, ['slices = [34, 35, 36, 37, 38, 39, 40, 41]'],
            "replicated causal input")
    if 'opcode = "fma"' in schedule or 'opcode = "fms"' in schedule:
        raise AssertionError("depthwise schedule still contains fused VXM opcodes")
    if "depthwise_conv1d_task" in schedule:
        raise AssertionError("depthwise task survived direct schedule lowering")
    if "qwen" in schedule.lower() or "deltanet" in schedule.lower():
        raise AssertionError("model-specific operation leaked into hardware IR")

    command = run(args.tool, args.input, args.output_dir / "conv.command.mlir",
                  "ftlpu-stablehlo-to-commands")
    require(command, ["ftlpu.command.mem_3d",
                      "ftlpu.command.vxm_run_2d",
                      'name = "depthwise_conv1d"'], "Command IR")
    if "depthwise_conv1d_task" in command:
        raise AssertionError("depthwise task survived command lowering")
    if "qwen" in command.lower() or "deltanet" in command.lower():
        raise AssertionError("model-specific operation leaked into command IR")

    print("generic DepthwiseConv1D lowered Kernel -> Tensor -> Stream -> "
          "Schedule -> Command")


if __name__ == "__main__":
    main()
