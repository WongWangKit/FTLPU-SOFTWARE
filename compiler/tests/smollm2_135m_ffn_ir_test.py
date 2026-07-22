#!/usr/bin/env python3
"""Lowers the SmolLM2-135M W8A16 FFN through every FTLPU IR layer."""

import argparse
import re
import shutil
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


def require(text: str, values: list[str], layer: str) -> None:
    missing = [value for value in values if value not in text]
    if missing:
        raise AssertionError(f"{layer} IR is missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stablehlo = args.input.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output_dir / "ffn.stablehlo.mlir")
    require(stablehlo, ["tensor<32x576xf16>", "tensor<576x1536xi8>",
                        "tensor<1536x576xi8>", "stablehlo.logistic"], "StableHLO")

    kernel = run(args.tool, args.input, args.output_dir / "ffn.kernel.mlir",
                 "ftlpu-stablehlo-to-kernel")
    require(kernel, ["ftlpu.kernel.ffn", "k = 576 : i64", "hidden = 1536 : i64",
                     "n = 576 : i64"], "Kernel")

    tensor = run(args.tool, args.input, args.output_dir / "ffn.tensor.mlir",
                 "ftlpu-stablehlo-to-tensor")
    require(tensor, ["ftlpu.tensor.ffn", "base_row = 0 : i64",
                     "base_row = 1728 : i64", "base_row = 3456 : i64",
                     'hemisphere = "both"', 'hemisphere = "west"',
                     "slices = [21, 22, 23, 29]",
                     'kind = "w8a16_mxm_weight_striped"'], "Tensor")

    stream = run(args.tool, args.input, args.output_dir / "ffn.stream.mlir",
                 "ftlpu-stablehlo-to-stream")
    require(stream, ["ftlpu.stream.ffn", "ftlpu.stream.dequantize",
                     'destination = "VXM.input"', 'source = "VXM.result"',
                     'destination = "MXM.weight"', "stream_count = 8 : i64",
                     "stream_count = 16 : i64", "stream_count = 4 : i64"], "Stream")

    schedule = run(args.tool, args.input, args.output_dir / "ffn.schedule.mlir",
                   "ftlpu-stablehlo-to-schedule")
    require(schedule, ["ftlpu.schedule.mem_read", "ftlpu.schedule.mxm_load",
                       "ftlpu.schedule.mxm_compute", "ftlpu.schedule.vxm",
                       "ftlpu.schedule.mem_accumulate", "ftlpu.schedule.mem_write",
                       'destination = "stream"'], "Schedule")
    swish_cycles = []
    for line in schedule.splitlines():
        if ("ftlpu.schedule.vxm" not in line
                or 'opcode = "negate"' not in line
                or 'lhs_kind = "stream_f32"' not in line):
            continue
        match = re.search(r"cycle = (\d+) : i64", line)
        if match:
            swish_cycles.append(int(match.group(1)))
    if not swish_cycles or min(swish_cycles) != 16469:
        raise AssertionError(
            f"Schedule expected tail SwiGLU start at cycle 16469, got {swish_cycles[:1]}")
    if swish_cycles[:10] != list(range(16469, 16479)):
        raise AssertionError(
            f"Schedule expected sequential tail SwiGLU events, got {swish_cycles[:10]}")

    command = run(args.tool, args.input, args.output_dir / "ffn.command.mlir",
                  "ftlpu-stablehlo-to-commands")
    require(command, ["ftlpu.command.binding", "ftlpu.command.mem",
                      "ftlpu.command.mxm", "ftlpu.command.vxm",
                      'kind = "fp16_pair_planar"'], "Command")


if __name__ == "__main__":
    main()
