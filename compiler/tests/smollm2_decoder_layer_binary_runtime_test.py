#!/usr/bin/env python3
"""Compiles and runs a complete standard-StableHLO decoder layer."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_phase(name: str, command: list[str], **kwargs: object) -> None:
    start = time.perf_counter()
    subprocess.run(command, check=True, **kwargs)
    print(f"{name}: {time.perf_counter() - start:.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mxm-execution",
        choices=("legacy", "vector"),
        default="legacy",
        help="select the decoder MXM execution policy",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="write the decoded runtime schedule CSV",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render the runtime schedule SVG; implies --trace",
    )
    args = parser.parse_args()
    if args.render:
        args.trace = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_ir = args.output_dir / "decoder_layer.schedule.mlir"
    command_ir = args.output_dir / "decoder_layer.command.mlir"
    binary = args.output_dir / "decoder_layer.ftlpu"
    trace = args.output_dir / "decoder_layer.runtime.csv"
    pipeline = args.output_dir / "decoder_layer.pipeline.svg"
    run_phase("stablehlo-to-schedule", [
        str(args.opt), "--input", str(args.input), "--output",
        str(schedule_ir), "--pipeline", "ftlpu-stablehlo-to-schedule",
        "--ffn-schedule", "tail",
        "--rmsnorm-strategy", "vxm-feedback",
        "--mxm-execution", args.mxm_execution,
        "--target-config", str(args.target_config),
    ])
    run_phase("schedule-to-command", [
        str(args.opt), "--input", str(schedule_ir), "--output",
        str(command_ir), "--pipeline", "ftlpu-verified-schedule-to-commands",
        "--target-config", str(args.target_config),
    ])
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
    for marker in (
        'weight_input_mode = "int8_dequant_bf16"',
        "ftlpu.command.mxm_dequant",
    ):
        if marker not in text:
            raise RuntimeError(
                f"vector decoder is missing Command IR marker: {marker}"
            )
    output_bindings = [
        line for line in text.splitlines()
        if "ftlpu.command.binding" in line
        and 'access = "output"' in line
        and "index = 0 : i64" in line
    ]
    if len(output_bindings) != 1 or (
        'kind = "fp16_mxm_distributed_16"' not in output_bindings[0]
        or 'element_type = "bf16"' not in output_bindings[0]
    ):
        raise RuntimeError(
            "decoder output must use the BF16 persistent distributed16 activation ABI"
        )
    run_phase("command-to-binary", [
        str(args.translate), "--input", str(command_ir),
        "--output", str(binary),
    ])
    environment = os.environ.copy()
    if args.trace:
        environment["FTLPU_SCHEDULE_TRACE"] = str(trace)
    else:
        environment.pop("FTLPU_SCHEDULE_TRACE", None)
    run_phase(
        "binary-runtime",
        [str(args.runtime_test), str(binary)],
        env=environment,
    )
    if args.render:
        renderer = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "render_decoder_layer_pipeline.py"
        )
        run_phase(
            "pipeline-render",
            [sys.executable, str(renderer), str(trace), str(pipeline)],
        )


if __name__ == "__main__":
    main()
