#!/usr/bin/env python3
"""Validates complete compiler-generated Attention on the runtime and CModel."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument(
        "--attention-schedule", choices=("tail", "fused"), default="tail")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_ir = args.output_dir / "attention.command.mlir"
    binary = args.output_dir / "attention.ftlpu"
    trace = args.output_dir / "attention.runtime.csv"
    pipeline = args.output_dir / "attention.pipeline.svg"
    subprocess.run([
        str(args.opt), "--input", str(args.input), "--output",
        str(command_ir), "--pipeline", "ftlpu-stablehlo-to-commands",
        "--attention-schedule", args.attention_schedule,
        "--target-config", str(args.target_config),
    ], check=True)
    command_text = command_ir.read_text(encoding="utf-8")
    required_ops = {
        "binding": "ftlpu.command.binding",
        "MEM": "ftlpu.command.mem",
        "MXM": "ftlpu.command.mxm",
        "VXM": "ftlpu.command.vxm",
        "SXM": "ftlpu.command.sxm",
    }
    missing = [name for name, op in required_ops.items() if op not in command_text]
    if missing:
        raise RuntimeError(
            "complete Attention Command IR is missing: " + ", ".join(missing))
    if 'opcode = "iw"' not in command_text or 'weight_column = 3' not in command_text:
        raise RuntimeError(
            "Attention Command IR is missing reverse-column QK IW loads")
    if 'opcode = "write_tap"' not in command_text:
        raise RuntimeError(
            "O-projection input staging is missing passive MEM taps")
    if args.attention_schedule == "fused":
        fused_selected = 'name = "softmax_fused"' in command_text
        tail_fallback = 'name = "softmax"' in command_text
        if not fused_selected and not tail_fallback:
            raise RuntimeError("Attention emitted neither Fused nor Tail softmax")
        if fused_selected and (
                'accumulator_destination = "stream"' not in command_text
                or 'accumulator_clear = true' not in command_text):
            raise RuntimeError(
                "fused Attention is missing QK accumulator stream-and-clear")
    subprocess.run([str(args.translate), "--input", str(command_ir),
                    "--output", str(binary)], check=True)
    environment = os.environ.copy()
    environment["FTLPU_SCHEDULE_TRACE"] = str(trace)
    runtime_result = subprocess.run(
        [str(args.runtime_test), str(binary)], env=environment)
    renderer = Path(__file__).resolve().parents[1] / "tools" / "render_attention_pipeline.py"
    subprocess.run([
        sys.executable, str(renderer), str(trace), str(pipeline),
    ], check=True)
    runtime_result.check_returncode()


if __name__ == "__main__":
    main()
