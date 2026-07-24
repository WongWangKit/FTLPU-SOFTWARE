#!/usr/bin/env python3
"""Compiles the SmolLM2-135M FFN and validates its CModel result."""

import argparse
import os
import re
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
    parser.add_argument("--ffn-schedule", choices=("tail", "fused"),
                        default="tail")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = args.output_dir / "ffn.command.mlir"
    binary = args.output_dir / "ffn.ftlpu"
    trace = args.output_dir / "ffn.runtime.csv"
    pipeline = args.output_dir / "ffn.pipeline.svg"
    subprocess.run([
        str(args.opt), "--input", str(args.input), "--output", str(commands),
        "--pipeline", "ftlpu-stablehlo-to-commands",
        "--ffn-schedule", args.ffn_schedule,
    ], check=True)
    subprocess.run([
        str(args.translate), "--input", str(commands), "--output", str(binary),
    ], check=True)
    environment = os.environ.copy()
    environment["FTLPU_SCHEDULE_TRACE"] = str(trace)
    runtime_result = subprocess.run([str(args.runtime_test), str(binary)], env=environment)
    renderer = Path(__file__).resolve().parents[1] / "tools" / "render_ffn_pipeline.py"
    source = args.input.read_text(encoding="utf-8")
    sequence_match = re.search(r"tensor<(\d+)x576xf16>", source)
    if not sequence_match:
        raise RuntimeError("cannot determine FFN sequence length from StableHLO input")
    subprocess.run([
        sys.executable, str(renderer), str(trace), str(pipeline),
        "--sequence-length", sequence_match.group(1),
    ], check=True)
    runtime_result.check_returncode()


if __name__ == "__main__":
    main()
