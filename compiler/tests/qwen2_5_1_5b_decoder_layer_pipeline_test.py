#!/usr/bin/env python3
"""Builds the heavyweight Qwen2.5-1.5B decoder-layer executable pipeline."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], phase: str) -> None:
    print(f"[{phase}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stablehlo = args.output_dir / "decoder_layer.stablehlo.mlir"
    stream = args.output_dir / "decoder_layer.stream.mlir"
    schedule = args.output_dir / "decoder_layer.schedule.mlir"
    command = args.output_dir / "decoder_layer.command.mlir"
    binary = args.output_dir / "decoder_layer.ftlpu"
    shutil.copyfile(args.input, stablehlo)

    common = [
        "--mxm-execution", "block8", "--ffn-schedule", "tail",
        "--target-config", str(args.target_config),
        "--rmsnorm-strategy", "vxm-feedback",
    ]
    run([
        str(args.opt), "--input", str(stablehlo), "--output", str(stream),
        "--pipeline", "ftlpu-stablehlo-to-stream", *common,
    ], "stablehlo-to-stream")
    run([
        str(args.opt), "--input", str(stream), "--output", str(schedule),
        "--pipeline", "ftlpu-stream-to-compressed-schedule", *common,
    ], "stream-to-schedule")

    schedule_markers = {
        "ftlpu.schedule.compressed", 'name = "qkv"',
        'name = "softmax"', 'name = "o_proj"',
        'name = "rmsnorm.feedback"', 'name = "ffn.down.block8"',
        'accumulator_destination = "stream"',
        'accumulator_clear = true',
    }
    with schedule.open(encoding="utf-8") as source:
        for line in source:
            schedule_markers = {
                marker for marker in schedule_markers if marker not in line
            }
            if "ftlpu.schedule.mxm_issue" not in line:
                continue
            match = re.search(r"accumulator_address = (\d+) : i64", line)
            if not match:
                continue
            address = int(match.group(1))
            limit = 128 if 'compute_mode = "block8"' in line else 1024
            if address >= limit:
                raise AssertionError(
                    f"MXM accumulator address {address} exceeds {limit} rows"
                )
    if schedule_markers:
        raise AssertionError(
            f"Schedule IR is missing {sorted(schedule_markers)}"
        )

    run([
        str(args.opt), "--input", str(schedule), "--output", str(command),
        "--pipeline", "ftlpu-verified-schedule-to-commands",
        "--target-config", str(args.target_config),
    ], "schedule-to-command")
    command_markers = {
        "ftlpu.command.binding", "ftlpu.command.mem",
        "ftlpu.command.mxm", "ftlpu.command.vxm", "ftlpu.command.sxm",
    }
    lowered_op = re.compile(r"^\s+(?:ftlpu\.schedule|ftlpu\.stream)\.")
    with command.open(encoding="utf-8") as source:
        for line in source:
            command_markers = {
                marker for marker in command_markers if marker not in line
            }
            if lowered_op.match(line):
                raise AssertionError(
                    f"Command IR retained a lowered op: {line.strip()}"
                )
    if command_markers:
        raise AssertionError(
            f"Command IR is missing {sorted(command_markers)}"
        )

    run([
        str(args.translate), "--input", str(command),
        "--output", str(binary),
    ], "command-to-binary")
    if binary.stat().st_size < 64:
        raise AssertionError("Qwen decoder-layer binary is unexpectedly small")
    print(
        f"Qwen2.5-1.5B decoder-layer executable: {binary} "
        f"({binary.stat().st_size} bytes)", flush=True,
    )


if __name__ == "__main__":
    main()
