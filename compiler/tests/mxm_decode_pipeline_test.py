#!/usr/bin/env python3
"""Compile and execute a single-token W8A16 projection through MXM Decode."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = args.output_dir / "mxm_decode.schedule.mlir"
    command = args.output_dir / "mxm_decode.command.mlir"
    binary = args.output_dir / "mxm_decode.ftlpu"

    subprocess.run([
        str(args.opt), "--input", str(args.input), "--output", str(schedule),
        "--pipeline", "ftlpu-stablehlo-to-schedule",
        "--mxm-execution", "vector",
    ], check=True)
    schedule_text = schedule.read_text(encoding="utf-8")
    for marker in (
        'kind = "w8a16_mxm_decode_streamed"',
        'kind = "fp16_mxm_decode_pair_planar"',
        'opcode = "decode_load_activation"',
        'opcode = "decode_stream_compute"',
        'decode_layout = "native_4x4"',
    ):
        if marker not in schedule_text:
            raise RuntimeError(f"decode schedule is missing marker: {marker}")

    subprocess.run([
        str(args.opt), "--input", str(schedule), "--output", str(command),
        "--pipeline", "ftlpu-verified-schedule-to-commands",
    ], check=True)
    command_text = command.read_text(encoding="utf-8")
    for marker in (
        'opcode = "decode_load_activation"',
        'opcode = "decode_stream_compute"',
        'opcode = "accumulator_read"',
    ):
        if marker not in command_text:
            raise RuntimeError(f"decode Command IR is missing marker: {marker}")

    subprocess.run([
        str(args.translate), "--input", str(command),
        "--output", str(binary),
    ], check=True)
    subprocess.run([str(args.runtime_test), str(binary)], check=True)


if __name__ == "__main__":
    main()
