#!/usr/bin/env python3
"""Checks that a non-default LPU target changes and survives lowering."""

import argparse
import subprocess
from pathlib import Path


def lower(tool: Path, source: Path, output: Path,
          target_config: Path | None = None) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(tool),
        "--input", str(source),
        "--output", str(output),
        "--pipeline", "ftlpu-stablehlo-to-schedule",
    ]
    if target_config is not None:
        command.extend(["--target-config", str(target_config)])
    subprocess.run(command, check=True)
    return output.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = lower(
        args.tool, args.input, args.output_dir / "default.schedule.mlir")
    explored = lower(
        args.tool, args.input, args.output_dir / "explored.schedule.mlir",
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
        "ftlpu.schedule.mxm_compute",
        "ftlpu.schedule.vxm",
    ]
    missing = [value for value in required if value not in explored]
    if missing:
        raise AssertionError(
            f"configured Schedule IR is missing: {missing}")
    if 'name = "lpu_32stream_v1"' not in baseline:
        raise AssertionError("default Schedule IR has no explicit 32-stream target")
    if baseline == explored:
        raise AssertionError("non-default target did not change Schedule IR")


if __name__ == "__main__":
    main()
