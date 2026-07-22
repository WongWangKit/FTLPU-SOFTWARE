#!/usr/bin/env python3
"""Verifies StableHLO-to-Stream IR routing for the 320x320 matmul."""

import argparse
import subprocess
from pathlib import Path


def run(tool, input_path, output_path, pipeline):
    subprocess.run(
        [
            str(tool),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--pipeline",
            pipeline,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.tool, args.input, args.output, "ftlpu-stablehlo-to-stream")
    text = args.output.read_text(encoding="utf-8")
    expected = [
        "ftlpu.stream.matmul",
        "unit_id = 0 : i64",
        "weight_buffer = 0 : i64",
        'direction = "east"',
        'source = "MEM"',
        'destination = "MXM.activation"',
        'destination = "MXM.weight"',
        'direction = "west"',
        'source = "MXM.result"',
        "stream_base = 0 : i64",
        "stream_base = 16 : i64",
        "stream_count = 16 : i64",
        "stream_count = 4 : i64",
        "stream_count = 4 : i64",
        "register_id = 1 : i64",
        "transport_latency = 13 : i64",
        "transport_latency = 5 : i64",
        "transport_latency = 2 : i64",
        "bytes = 102400 : i64",
        "bytes = 409600 : i64",
    ]
    missing = [item for item in expected if item not in text]
    if missing:
        raise AssertionError(f"missing Stream IR details: {missing}")
    if text.count("ftlpu.stream.route") != 3:
        raise AssertionError("matmul must lower to exactly three stream routes")
    identity_counts = {
        "source_unit_id = -1 : i64": 2,
        "source_unit_id = 0 : i64": 1,
        "destination_unit_id = 0 : i64": 2,
        "destination_unit_id = -1 : i64": 1,
    }
    for attribute, count in identity_counts.items():
        if text.count(attribute) != count:
            raise AssertionError(
                f"expected {count} occurrences of {attribute}, got {text.count(attribute)}"
            )
    if "lifetime_start" in text or "lifetime_end" in text:
        raise AssertionError("logical lifetime stages must not leak into Stream IR")
    forbidden = ["stablehlo.dot_general", "ftlpu.kernel.matmul", "ftlpu.tensor.matmul"]
    present = [item for item in forbidden if item in text]
    if present:
        raise AssertionError(f"operations remained after Stream lowering: {present}")

    roundtrip = args.output.with_suffix(".roundtrip.mlir")
    run(args.tool, args.output, roundtrip, "ftlpu-stablehlo-to-kernel")
    if roundtrip.read_text(encoding="utf-8").count("ftlpu.stream.route") != 3:
        raise AssertionError("Stream IR did not survive parse/print roundtrip")


if __name__ == "__main__":
    main()
