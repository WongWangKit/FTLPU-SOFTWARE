#!/usr/bin/env python3
"""Verifies attention lowers to a physically planned Tensor task DAG."""

import argparse
import re
import subprocess
from pathlib import Path


def run(tool: Path, input_path: Path, output_path: Path, pipeline: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.tool, args.input, args.output, "ftlpu-stablehlo-to-tensor")
    text = args.output.read_text(encoding="utf-8")

    expected_counts = {
        "ftlpu.tensor.projection_task": 4,
        "ftlpu.tensor.rope_task": 2,
        "ftlpu.tensor.batch_matmul_task": 2,
        "ftlpu.tensor.softmax_task": 1,
        "ftlpu.tensor.transpose_task": 2,
    }
    for operation, expected in expected_counts.items():
        actual = text.count(operation)
        if actual != expected:
            raise AssertionError(
                f"expected {expected} {operation} operations, got {actual}"
            )

    forbidden = (
        "ftlpu.tensor.attention",
        "ftlpu.kernel.attention",
        "stablehlo.",
    )
    present = [operation for operation in forbidden if operation in text]
    if present:
        raise AssertionError(f"operations remained after Tensor lowering: {present}")

    graph_pattern = re.compile(
        r"(%\w+) = ftlpu\.tensor\.projection_task %arg0, %arg1.*?"
        r"(%\w+) = ftlpu\.tensor\.projection_task %arg0, %arg2.*?"
        r"(%\w+) = ftlpu\.tensor\.projection_task %arg0, %arg3.*?"
        r"(%\w+) = ftlpu\.tensor\.rope_task \1.*?"
        r"(%\w+) = ftlpu\.tensor\.rope_task \2.*?"
        r"(%\w+) = ftlpu\.tensor\.batch_matmul_task \4, \5.*?"
        r"(%\w+) = ftlpu\.tensor\.softmax_task \6.*?"
        r"(%\w+) = ftlpu\.tensor\.transpose_task \7.*?"
        r"(%\w+) = ftlpu\.tensor\.transpose_task \3.*?"
        r"(%\w+) = ftlpu\.tensor\.batch_matmul_task \8, \9.*?"
        r"(%\w+) = ftlpu\.tensor\.projection_task \10, %arg4",
        re.DOTALL,
    )
    if not graph_pattern.search(text):
        raise AssertionError("Tensor attention tasks are not connected as an SSA DAG")

    plan_entries = (
        "input",
        "query_weight",
        "key_weight",
        "value_weight",
        "output_weight",
        "query",
        "key",
        "value",
        "score",
        "score_mxm1",
        "exp",
        "exp_mxm1",
        "causal_mask",
        "causal_mask_mxm1",
        "probability",
        "probability_mxm1",
        "probability_pack",
        "probability_diagonal",
        "rope",
        "context",
        "result",
    )
    for entry in plan_entries:
        count = text.count(f"{entry} =")
        if count != 1:
            raise AssertionError(
                f"physical plan entry {entry} must have one task owner, got {count}"
            )
    if text.count("memory_plan =") != 11:
        raise AssertionError("every Tensor attention task must expose its local sub-plan")

    required_layouts = (
        "slices = [28, 29, 30, 31]",
        'kind = "fp16_value_x16"',
        "base_row = 7800 : i64",
        'kind = "fp16_probability_x16"',
        "base_row = 6000 : i64",
        'kind = "fp16_probability_diagonal"',
        "base_row = 7000 : i64",
        'kind = "fp32_causal_mask_tile"',
        "base_row = 8128 : i64",
    )
    missing = [item for item in required_layouts if item not in text]
    if missing:
        raise AssertionError(f"Tensor attention physical layout is missing: {missing}")

    roundtrip = args.output.with_suffix(".roundtrip.mlir")
    run(args.tool, args.output, roundtrip, "ftlpu-stablehlo-to-kernel")
    roundtrip_text = roundtrip.read_text(encoding="utf-8")
    for operation, expected in expected_counts.items():
        if roundtrip_text.count(operation) != expected:
            raise AssertionError(f"{operation} did not survive parse/print roundtrip")


if __name__ == "__main__":
    main()
