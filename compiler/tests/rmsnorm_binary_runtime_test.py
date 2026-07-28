#!/usr/bin/env python3
"""Lowers generic StableHLO RMSNorm and checks the binary on CModel."""

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
    parser.add_argument(
        "--strategy",
        choices=("vxm-square-mxm-reduce", "vxm-feedback"),
        default="vxm-square-mxm-reduce",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command_ir = args.output_dir / "rmsnorm.command.mlir"
    binary = args.output_dir / "rmsnorm.ftlpu"
    subprocess.run([
        str(args.opt), "--input", str(args.input), "--output",
        str(command_ir), "--pipeline", "ftlpu-stablehlo-to-commands",
        "--rmsnorm-strategy", args.strategy,
    ], check=True)
    text = command_ir.read_text(encoding="utf-8")
    for operation in (
        "ftlpu.command.binding",
        "ftlpu.command.mem",
        "ftlpu.command.vxm",
    ):
        if operation not in text:
            raise RuntimeError(f"RMSNorm Command IR is missing {operation}")
    for opcode in ("square", "sqrt", "divide", "multiply"):
        if f'opcode = "{opcode}"' not in text:
            raise RuntimeError(f"RMSNorm Command IR is missing {opcode}")
    if args.strategy == "vxm-feedback":
        if "source_streams = [0]" in text:
            raise RuntimeError("feedback RMSNorm emitted a legacy narrow SXM command")
    elif "ftlpu.command.mxm" not in text:
        raise RuntimeError("MXM-reduce RMSNorm Command IR is missing MXM")

    subprocess.run([
        str(args.translate), "--input", str(command_ir),
        "--output", str(binary),
    ], check=True)
    subprocess.run([
        str(args.runtime_test), str(binary), args.strategy,
    ], check=True)


if __name__ == "__main__":
    main()
