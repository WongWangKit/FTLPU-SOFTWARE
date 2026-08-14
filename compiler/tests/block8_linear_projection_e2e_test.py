#!/usr/bin/env python3
"""Compile a generic W8A16 projection and run its Block8 binary on CModel."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--block8-target-config", type=Path, required=True)
    parser.add_argument("--legacy-target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = args.output_dir / "block8_linear_projection.command.mlir"
    binary = args.output_dir / "block8_linear_projection.ftlpu"
    subprocess.run(
        [
            str(args.opt),
            "--input",
            str(args.input),
            "--output",
            str(command),
            "--pipeline",
            "ftlpu-stablehlo-to-commands",
            "--target-config",
            str(args.block8_target_config),
        ],
        check=True,
    )
    text = command.read_text(encoding="utf-8")
    required = (
        'weight_input_mode = "int8_dequant_bf16"',
        'compute_mode = "block8"',
        "ftlpu.command.mxm_dequant",
        'kind = "fp16_mxm_block8_distributed_16"',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"command IR misses selected strategy: {missing}")

    legacy_command = args.output_dir / "legacy_linear_projection.command.mlir"
    legacy_args = [
        str(args.opt),
        "--input",
        str(args.input),
        "--pipeline",
        "ftlpu-stablehlo-to-commands",
        "--target-config",
        str(args.legacy_target_config),
    ]
    subprocess.run(
        legacy_args + ["--output", str(legacy_command),
                       "--mxm-execution", "auto"],
        check=True,
    )
    legacy_text = legacy_command.read_text(encoding="utf-8")
    if ('compute_mode = "vector"' not in legacy_text
            or 'compute_mode = "block8"' in legacy_text):
        raise RuntimeError(
            "target without Block8 capability did not select Vector compute"
        )
    forced = subprocess.run(
        legacy_args + [
            "--output",
            str(args.output_dir / "unsupported_block8.command.mlir"),
            "--mxm-execution",
            "block8",
        ],
        capture_output=True,
        text=True,
    )
    if forced.returncode == 0:
        raise RuntimeError(
            "Block8 policy enabled a capability missing from the target"
        )
    subprocess.run(
        [
            str(args.translate),
            "--input",
            str(command),
            "--output",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(args.runtime_test), str(binary)], check=True)


if __name__ == "__main__":
    main()
