#!/usr/bin/env python3
"""Lowers generic StableHLO RMSNorm and checks the binary on CModel."""

import argparse
import re
import subprocess
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
        "--target-config", str(args.target_config),
    ], check=True)
    text = command_ir.read_text(encoding="utf-8")
    for operation in (
        "ftlpu.command.binding",
        "ftlpu.command.mem",
        "ftlpu.command.vxm",
    ):
        if operation not in text:
            raise RuntimeError(f"RMSNorm Command IR is missing {operation}")
    if args.strategy == "vxm-feedback":
        for marker in (
            'kind = "fp16_vxm_row_parallel_8"',
            'opcode = "multiply"',
            'opcode = "rsqrt"',
            "accumulator_reset = true",
            "accumulator_write = true",
            "accumulator_emit = false",
            "local_scalar_write = true",
            "chain_depth = 2",
            "chain_depth = 4",
        ):
            if marker not in text:
                raise RuntimeError(
                    f"feedback RMSNorm Command IR is missing {marker}")
        if 'opcode = "square"' in text or 'opcode = "sqrt"' in text:
            raise RuntimeError("feedback RMSNorm emitted a non-hardware VXM opcode")
        mem_queues = [
            int(value)
            for value in re.findall(
                r'ftlpu\.command\.mem \{[^\n]*?queue = (\d+) : i64', text
            )
        ]
        if not any(queue % 2 == 1 for queue in mem_queues):
            raise RuntimeError(
                "feedback RMSNorm Command IR does not use SRAM bank 1")
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
