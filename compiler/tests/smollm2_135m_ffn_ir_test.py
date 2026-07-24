#!/usr/bin/env python3
"""Lowers the SmolLM2-135M W8A16 FFN through every FTLPU IR layer."""

import argparse
from collections import defaultdict
import re
import shutil
import subprocess
from pathlib import Path


def run(tool: Path, source: Path, output: Path, pipeline: str,
        ffn_schedule: str) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(tool), "--input", str(source), "--output", str(output),
         "--pipeline", pipeline, "--ffn-schedule", ffn_schedule],
        check=True,
    )
    return output.read_text(encoding="utf-8")


def require(text: str, values: list[str], layer: str) -> None:
    missing = [value for value in values if value not in text]
    if missing:
        raise AssertionError(f"{layer} IR is missing: {missing}")


def verify_vxm_queue_slots(schedule: str) -> None:
    slots: dict[tuple[int, int], list[int]] = defaultdict(list)
    for line_number, line in enumerate(schedule.splitlines(), 1):
        if "ftlpu.schedule.vxm" not in line:
            continue
        attrs = {
            key: int(value)
            for key, value in re.findall(
                r"(cycle|queue|repeat_count|repeat_interval) = (\d+) : i64",
                line,
            )
        }
        if len(attrs) != 4:
            raise AssertionError(
                f"cannot decode VXM scheduling attributes on line {line_number}")
        for repeat in range(attrs["repeat_count"]):
            cycle = attrs["cycle"] + repeat * attrs["repeat_interval"]
            slots[(attrs["queue"], cycle)].append(line_number)
    collisions = {
        slot: lines for slot, lines in slots.items() if len(lines) > 1
    }
    if collisions:
        sample = list(collisions.items())[:5]
        raise AssertionError(f"VXM queue/cycle collisions: {sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffn-schedule", choices=("tail", "fused"),
                        default="tail")
    args = parser.parse_args()

    stablehlo = args.input.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output_dir / "ffn.stablehlo.mlir")
    require(stablehlo, ["tensor<32x576xf16>", "tensor<576x1536xi8>",
                        "tensor<1536x576xi8>", "stablehlo.logistic"], "StableHLO")

    kernel = run(args.tool, args.input, args.output_dir / "ffn.kernel.mlir",
                 "ftlpu-stablehlo-to-kernel", args.ffn_schedule)
    require(kernel, ["ftlpu.kernel.matmul", "ftlpu.kernel.swish",
                     "ftlpu.kernel.elementwise", "k = 576 : i64",
                     "n = 1536 : i64", "n = 576 : i64"], "Kernel")
    if "ftlpu.kernel.ffn" in kernel:
        raise AssertionError("FFN must be decomposed in public Kernel IR")
    if kernel.count("ftlpu.kernel.matmul") != 3:
        raise AssertionError("FFN must contain gate, up, and down matmuls")

    tensor = run(args.tool, args.input, args.output_dir / "ffn.tensor.mlir",
                 "ftlpu-stablehlo-to-tensor", args.ffn_schedule)
    require(tensor, ["ftlpu.tensor.matmul_task", "ftlpu.tensor.swish_task",
                     "ftlpu.tensor.elementwise_task",
                     "result_allocations = []", "base_row = 0 : i64",
                     "base_row = 1728 : i64", "base_row = 3456 : i64",
                     'hemisphere = "both"', 'hemisphere = "west"',
                     "slices = [21, 22, 23, 29]",
                     'kind = "w8a16_mxm_weight_striped"',
                     'kind = "w8a16_mxm_weight_wave_striped"'], "Tensor")
    if "ftlpu.tensor.ffn" in tensor:
        raise AssertionError("Tensor IR must not contain the legacy compound FFN op")

    stream = run(args.tool, args.input, args.output_dir / "ffn.stream.mlir",
                 "ftlpu-stablehlo-to-stream", args.ffn_schedule)
    require(stream, ["ftlpu.stream.matmul_task", "ftlpu.stream.swish_task",
                     "ftlpu.stream.elementwise_task",
                     'kind = "multiply"', 'kind = "add_quant"',
                     "ftlpu.stream.dequantize",
                     'destination = "VXM.input"', 'source = "VXM.result"',
                     'destination = "MXM.weight"', "stream_count = 8 : i64",
                     "stream_count = 16 : i64", "stream_count = 4 : i64"], "Stream")
    if "ftlpu.stream.ffn" in stream:
        raise AssertionError("Stream IR must not contain the legacy compound FFN op")

    schedule = run(args.tool, args.input, args.output_dir / "ffn.schedule.mlir",
                   "ftlpu-stablehlo-to-schedule", args.ffn_schedule)
    require(schedule, ["ftlpu.schedule.mem_read", "ftlpu.schedule.mxm_load",
                       "ftlpu.schedule.mxm_compute", "ftlpu.schedule.vxm",
                       "ftlpu.schedule.mem_accumulate", "ftlpu.schedule.mem_write",
                       'destination = "stream"'], "Schedule")
    verify_vxm_queue_slots(schedule)
    swish_cycles = []
    for line in schedule.splitlines():
        if ("ftlpu.schedule.vxm" not in line
                or 'opcode = "negate"' not in line
                or 'lhs_kind = "stream_f32"' not in line):
            continue
        match = re.search(r"cycle = (\d+) : i64", line)
        if match:
            swish_cycles.append(int(match.group(1)))
    if args.ffn_schedule == "tail":
        if not swish_cycles or min(swish_cycles) != 13883:
            raise AssertionError(
                "Schedule expected tail SwiGLU start at cycle 13883, "
                f"got {swish_cycles[:1]}")
        if swish_cycles[:10] != list(range(13883, 13893)):
            raise AssertionError(
                "Schedule expected sequential tail SwiGLU events, "
                f"got {swish_cycles[:10]}")
    else:
        if not swish_cycles or min(swish_cycles) >= 13883:
            raise AssertionError(
                "Fused schedule did not overlap SwiGLU with projection: "
                f"{swish_cycles[:1]}")
        if 'destination = "stream"' not in schedule:
            raise AssertionError(
                "Fused schedule did not emit accumulator stream+clear")
        require(schedule, ['kind = "fp32_swiglu_temp_byte"',
                           "output_stream_base = 8 : i64",
                           "output_stream_base = 16 : i64",
                           "stream_base = 8 : i64",
                           "stream_base = 16 : i64"],
                "Fused Schedule")

    command = run(args.tool, args.input, args.output_dir / "ffn.command.mlir",
                  "ftlpu-stablehlo-to-commands", args.ffn_schedule)
    require(command, ["ftlpu.command.binding", "ftlpu.command.mem",
                      "ftlpu.command.mxm", "ftlpu.command.vxm",
                      'kind = "fp16_pair_planar"',
                      'hemisphere = "both"'], "Command")


if __name__ == "__main__":
    main()
