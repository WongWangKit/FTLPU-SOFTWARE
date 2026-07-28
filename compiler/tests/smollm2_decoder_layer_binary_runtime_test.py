#!/usr/bin/env python3
"""Compiles and runs a complete standard-StableHLO decoder layer."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_ir = args.output_dir / "decoder_layer.schedule.mlir"
    command_ir = args.output_dir / "decoder_layer.command.mlir"
    binary = args.output_dir / "decoder_layer.ftlpu"
    trace = args.output_dir / "decoder_layer.runtime.csv"
    pipeline = args.output_dir / "decoder_layer.pipeline.svg"
    subprocess.run([
        str(args.opt), "--input", str(args.input), "--output",
        str(schedule_ir), "--pipeline", "ftlpu-stablehlo-to-schedule",
        "--rmsnorm-strategy", "vxm-feedback",
        "--target-config", str(args.target_config),
    ], check=True)
    subprocess.run([
        str(args.opt), "--input", str(schedule_ir), "--output",
        str(command_ir), "--pipeline", "ftlpu-schedule-to-commands",
        "--target-config", str(args.target_config),
    ], check=True)
    text = command_ir.read_text(encoding="utf-8")
    for operation in (
        "ftlpu.command.binding",
        "ftlpu.command.mem",
        "ftlpu.command.mxm",
        "ftlpu.command.vxm",
        "ftlpu.command.sxm",
    ):
        if operation not in text:
            raise RuntimeError(f"decoder layer is missing {operation}")
    output_bindings = [
        line for line in text.splitlines()
        if "ftlpu.command.binding" in line
        and 'access = "output"' in line
        and "index = 0 : i64" in line
    ]
    if len(output_bindings) != 1 or (
        'kind = "fp16_mxm_distributed_16"' not in output_bindings[0]
    ):
        raise RuntimeError(
            "decoder output must use the persistent distributed16 activation ABI"
        )
    subprocess.run([
        str(args.translate), "--input", str(command_ir),
        "--output", str(binary),
    ], check=True)
    environment = os.environ.copy()
    environment["FTLPU_SCHEDULE_TRACE"] = str(trace)
    subprocess.run(
        [str(args.runtime_test), str(binary)],
        env=environment,
        check=True,
    )
    renderer = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "render_decoder_layer_pipeline.py"
    )
    subprocess.run(
        [sys.executable, str(renderer), str(trace), str(pipeline)],
        check=True,
    )


if __name__ == "__main__":
    main()
