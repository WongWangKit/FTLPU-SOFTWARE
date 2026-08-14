#!/usr/bin/env python3
"""Checks that a non-model-specific W8A16 FFN lowers through every IR layer."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def lower(tool: Path, source: Path, output: Path, pipeline: str) -> str:
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
    source = args.input.read_text(encoding="utf-8")
    bf16 = "xbf16>" in source

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output_dir / "ffn.stablehlo.mlir")

    kernel = lower(args.tool, args.input, args.output_dir / "ffn.kernel.mlir",
                   "ftlpu-stablehlo-to-kernel")
    require(kernel, ["ftlpu.kernel.matmul", "ftlpu.kernel.swish",
                     "ftlpu.kernel.elementwise", "k = 64 : i64",
                     "n = 128 : i64", "n = 64 : i64"], "Kernel")
    if "ftlpu.kernel.ffn" in kernel:
        raise AssertionError("FFN must be decomposed in public Kernel IR")
    if kernel.count("ftlpu.kernel.matmul") != 3:
        raise AssertionError("FFN must contain gate, up, and down matmuls")

    tensor = lower(args.tool, args.input, args.output_dir / "ffn.tensor.mlir",
                   "ftlpu-stablehlo-to-tensor")
    down_weight_kind = (
        'kind = "w8a16_block8_weight_wave_striped"'
        if bf16 else 'kind = "w8a16_mxm_weight_wave_striped"'
    )
    require(tensor, ["ftlpu.tensor.matmul_task", "ftlpu.tensor.swish_task",
                     "ftlpu.tensor.elementwise_task",
                     'kind = "w8a16_mxm_weight_striped"',
                     down_weight_kind,
                     'hemisphere = "both"'], "Tensor")
    if "ftlpu.tensor.ffn" in tensor:
        raise AssertionError("Tensor IR must not contain the legacy compound FFN op")

    stream = lower(args.tool, args.input, args.output_dir / "ffn.stream.mlir",
                   "ftlpu-stablehlo-to-stream")
    require(stream, ["ftlpu.stream.matmul_task", "ftlpu.stream.swish_task",
                     "ftlpu.stream.elementwise_task",
                     'kind = "multiply"', 'kind = "add_quant"',
                     'destination = "MXM.weight"',
                     "stream_count = 16 : i64",
                     "result_stream_counts = [4]"], "Stream")
    if not bf16:
        require(stream, ["ftlpu.stream.dequantize",
                         "stream_count = 4 : i64"], "Stream")
    if "ftlpu.stream.ffn" in stream:
        raise AssertionError("Stream IR must not contain the legacy compound FFN op")

    schedule = lower(args.tool, args.input, args.output_dir / "ffn.schedule.mlir",
                     "ftlpu-stablehlo-to-schedule")
    schedule_ops = (
        ["ftlpu.schedule.mem_transfer", "ftlpu.schedule.mxm_issue",
         "ftlpu.schedule.mxm_dequant", "ftlpu.schedule.vxm"]
        if bf16 else
        ["ftlpu.schedule.mem_read", "ftlpu.schedule.mxm_load",
         "ftlpu.schedule.mxm_compute", "ftlpu.schedule.vxm",
         "ftlpu.schedule.mem_write"]
    )
    require(schedule, schedule_ops + ["weight_buffer = 1 : i64"],
            "Schedule")
    if bf16:
        require(schedule, ['data_format = "bf16"',
                           'cast_target = "bf16"'], "BF16 Schedule")

    command = lower(args.tool, args.input, args.output_dir / "ffn.command.mlir",
                    "ftlpu-stablehlo-to-commands")
    require(command, ["ftlpu.command.binding", "ftlpu.command.mem",
                      "ftlpu.command.mxm", "ftlpu.command.vxm"], "Command")
    if bf16:
        require(command, ['element_type = "bf16"',
                          'data_format = "bf16"',
                          'cast_target = "bf16"'], "BF16 Command")
    iw_columns = []
    for line in command.splitlines():
        if 'opcode = "iw"' not in line:
            continue
        column = re.search(r"weight_column = (\d+) : i64", line)
        wave_count = re.search(r"wave_count = (\d+) : i64", line)
        wave_stride = re.search(
            r"wave_weight_column_stride = (-?\d+) : i64", line)
        if not column:
            continue
        count = int(wave_count.group(1)) if wave_count else 1
        stride = int(wave_stride.group(1)) if wave_stride else 0
        iw_columns.extend(
            int(column.group(1)) + wave * stride for wave in range(count))
        if len(iw_columns) >= 4:
            break
    expected_columns = [0, 1, 2, 3] if bf16 else [3, 2, 1, 0]
    if iw_columns[:4] != expected_columns:
        raise AssertionError(
            "MXM IW pulses do not follow the selected load mode's physical "
            f"column order {expected_columns}; got {iw_columns[:4]}")


if __name__ == "__main__":
    main()
