#!/usr/bin/env python3
"""Build and validate the Qwen3-0.6B seq32 prefill-layer pipeline."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], phase: str) -> None:
    print(f"[{phase}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--weight-bank", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stablehlo = args.output_dir / "decoder_layer.stablehlo.mlir"
    stream = args.output_dir / "decoder_layer.stream.mlir"
    schedule = args.output_dir / "decoder_layer.schedule.mlir"
    binary = args.output_dir / "decoder_layer.ftlpu"
    shutil.copyfile(args.input, stablehlo)

    common = [
        "--mxm-execution", "vector",
        "--ffn-schedule", "fused",
        "--target-config", str(args.target_config),
        "--weight-bank", str(args.weight_bank),
        "--icu-macro-schedule",
    ]
    run([
        str(args.opt), "--input", str(stablehlo), "--output", str(stream),
        "--pipeline", "ftlpu-stablehlo-to-stream", *common,
    ], "stablehlo-to-stream")
    run([
        str(args.opt), "--input", str(stream), "--output", str(schedule),
        "--pipeline", "ftlpu-stream-to-compressed-schedule", *common,
    ], "stream-to-compressed-schedule")

    text = schedule.read_text(encoding="utf-8")
    required = [
        "ftlpu.schedule.compressed",
        'name = "query_norm_weight"',
        'name = "key_norm_weight"',
        'name = "qkv"',
        'name = "qk"',
        'name = "softmax"',
        'name = "pv"',
        'name = "o_proj"',
        'name = "rmsnorm.feedback"',
        'name = "ffn.down.vector"',
        "tensor<32x2048xbf16>",
        "tensor<2048x1024xi8>",
    ]
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise AssertionError(f"schedule is missing Qwen3 markers: {missing}")
    for unexpected in ("query_bias", "key_bias", "value_bias"):
        if unexpected in text:
            raise AssertionError(f"Qwen3 schedule contains {unexpected}")

    down_binding = next(
        (
            line
            for line in text.splitlines()
            if "ftlpu.schedule.mem_read %arg11" in line
            and "binding_placement" in line
        ),
        None,
    )
    if down_binding is None:
        raise AssertionError("schedule is missing the Down weight binding")
    banks_match = re.search(r"page_banks = \[([^\]]+)\]", down_binding)
    rows_match = re.search(r"page_base_rows = \[([^\]]+)\]", down_binding)
    if banks_match is None or rows_match is None:
        raise AssertionError("Down weight binding is missing streaming page slots")
    page_banks = [int(value.strip()) for value in banks_match.group(1).split(",")]
    page_rows = [int(value.strip()) for value in rows_match.group(1).split(",")]
    expected_banks = [
        (args.weight_bank + page) % 2 for page in range(len(page_banks))
    ]
    if len(page_banks) != 8 or page_banks != expected_banks:
        raise AssertionError(
            f"Down tiles do not alternate reusable SRAM banks: {page_banks}"
        )
    if page_rows != [0] * len(page_banks):
        raise AssertionError(
            f"Down tiles do not reuse one row slot per bank: {page_rows}"
        )

    timelines: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        if "ftlpu.schedule.timeline" not in line:
            continue
        name = re.search(r'name = "([^"]+)"', line)
        start = re.search(r"start = (\d+) : i64", line)
        end = re.search(r"end = (\d+) : i64", line)
        if name and start and end:
            timelines[name.group(1)] = (int(start.group(1)), int(end.group(1)))
    ordered = ["qkv", "qk", "softmax", "pv", "o_proj", "ffn.down.vector"]
    for previous, current in zip(ordered, ordered[1:]):
        if timelines[previous][1] > timelines[current][0]:
            raise AssertionError(
                f"overlapping stage timelines: {previous} and {current}"
            )

    run([
        str(args.compile), "--input", str(schedule),
        "--output", str(binary), "--input-stage", "schedule",
        "--target-config", str(args.target_config),
        "--mxm-execution", "vector",
        "--weight-bank", str(args.weight_bank),
        "--icu-macro-schedule",
    ], "compressed-schedule-to-binary")
    if binary.stat().st_size < 64:
        raise AssertionError("Qwen3 decoder-layer binary is unexpectedly small")
    print(
        f"Qwen3-0.6B seq32 prefill executable: {binary} "
        f"({binary.stat().st_size} bytes)",
        flush=True,
    )


if __name__ == "__main__":
    main()
