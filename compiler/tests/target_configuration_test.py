#!/usr/bin/env python3
"""Checks that a non-default LPU target changes and survives lowering."""

import argparse
import json
import subprocess
from pathlib import Path


def lower(tool: Path, source: Path, output: Path,
          target_config: Path | None = None) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(tool),
        "--input", str(source),
        "--output", str(output),
        "--pipeline", "ftlpu-stablehlo-to-kernel",
    ]
    if target_config is not None:
        command.extend(["--target-config", str(target_config)])
    subprocess.run(command, check=True)
    return output.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = lower(
        args.tool, args.input, args.output_dir / "default.kernel.mlir")
    hardware = lower(
        args.tool, args.input, args.output_dir / "hardware.kernel.mlir",
        args.hardware_config)
    explored = lower(
        args.tool, args.input, args.output_dir / "explored.kernel.mlir",
        args.target_config)

    required = [
        "ftlpu.target",
        'name = "lpu_exploration_40stream"',
        'abi = "0x',
        "streams_per_direction = 40 : i64",
        "encoded_streams = 80 : i64",
        "mxm_pipeline_rows = 6 : i64",
        "vxm_weight_to_iw_latency = 10 : i64",
        "swiglu_write_latency = 11 : i64",
        "ftlpu.kernel.matmul",
        "ftlpu.kernel.swish",
        "ftlpu.kernel.elementwise",
    ]
    missing = [value for value in required if value not in explored]
    if missing:
        raise AssertionError(
            f"configured Kernel IR is missing: {missing}")
    if 'name = "ftlpu-lpu32"' not in baseline:
        raise AssertionError("default Kernel IR has no explicit 32-stream target")
    config = json.loads(args.hardware_config.read_text(encoding="utf-8"))
    shared_fields = [
        f"sram_depth_rows = {config['mem']['rows_per_bank']} : i64",
        f"mxm_accumulator_blocks = {config['mxm']['accum_contexts']} : i64",
    ]
    for field in shared_fields:
        if field not in baseline or field not in hardware:
            raise AssertionError(
                f"compiler target does not inherit shared hardware JSON field: {field}")
    if baseline == explored:
        raise AssertionError("non-default target did not change Schedule IR")


if __name__ == "__main__":
    main()
