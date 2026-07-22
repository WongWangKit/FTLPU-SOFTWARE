#!/usr/bin/env python3
"""Checks that a non-model-specific W8A16 FFN lowers through every IR layer."""

import argparse
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output_dir / "ffn.stablehlo.mlir")

    kernel = lower(args.tool, args.input, args.output_dir / "ffn.kernel.mlir",
                   "ftlpu-stablehlo-to-kernel")
    require(kernel, ["ftlpu.kernel.ffn", "k = 64 : i64", "hidden = 128 : i64",
                     "n = 64 : i64"], "Kernel")

    tensor = lower(args.tool, args.input, args.output_dir / "ffn.tensor.mlir",
                   "ftlpu-stablehlo-to-tensor")
    require(tensor, ["ftlpu.tensor.ffn", 'kind = "w8a16_mxm_weight_striped"',
                     'hemisphere = "both"'], "Tensor")

    stream = lower(args.tool, args.input, args.output_dir / "ffn.stream.mlir",
                   "ftlpu-stablehlo-to-stream")
    require(stream, ["ftlpu.stream.ffn", "ftlpu.stream.dequantize",
                     'destination = "MXM.weight"', "stream_count = 16 : i64",
                     "stream_count = 4 : i64"], "Stream")

    schedule = lower(args.tool, args.input, args.output_dir / "ffn.schedule.mlir",
                     "ftlpu-stablehlo-to-schedule")
    require(schedule, ["ftlpu.schedule.mem_read", "ftlpu.schedule.mxm_load",
                       "ftlpu.schedule.mxm_compute", "ftlpu.schedule.vxm",
                       "ftlpu.schedule.mem_write", "weight_buffer = 1 : i64"],
            "Schedule")

    command = lower(args.tool, args.input, args.output_dir / "ffn.command.mlir",
                    "ftlpu-stablehlo-to-commands")
    require(command, ["ftlpu.command.binding", "ftlpu.command.mem",
                      "ftlpu.command.mxm", "ftlpu.command.vxm"], "Command")


if __name__ == "__main__":
    main()
