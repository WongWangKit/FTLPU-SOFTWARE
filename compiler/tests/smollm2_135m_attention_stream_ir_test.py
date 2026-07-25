#!/usr/bin/env python3
"""Verifies attention lowers to a primitive, SSA-connected Stream task graph."""

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
    run(args.tool, args.input, args.output, "ftlpu-stablehlo-to-stream")
    text = args.output.read_text(encoding="utf-8")

    expected_counts = {
        "ftlpu.stream.projection_task": 4,
        "ftlpu.stream.rope_task": 2,
        "ftlpu.stream.batch_matmul_task": 2,
        "ftlpu.stream.softmax_task": 1,
        "ftlpu.stream.transpose_task": 2,
    }
    for operation, expected in expected_counts.items():
        actual = text.count(operation)
        if actual != expected:
            raise AssertionError(
                f"expected {expected} {operation} operations, got {actual}"
            )
    if text.count("memory_plan =") != 1:
        raise AssertionError("the shared physical memory plan must have one owner")
    if text.count('role = "') != 24:
        raise AssertionError("attention routes must be partitioned without duplication")

    forbidden = (
        "ftlpu.stream.attention",
        "ftlpu.tensor.attention",
        "ftlpu.kernel.attention",
        "stablehlo.",
    )
    present = [operation for operation in forbidden if operation in text]
    if present:
        raise AssertionError(f"operations remained after Stream lowering: {present}")

    for kind in (
        "query",
        "key",
        "value",
        "output",
        "qk",
        "pv",
        "probability",
    ):
        if f'kind = "{kind}"' not in text:
            raise AssertionError(f"primitive attention graph is missing kind {kind}")

    graph_pattern = re.compile(
        r"(%\w+) = ftlpu\.stream\.projection_task %arg0, %arg1.*?"
        r"(%\w+) = ftlpu\.stream\.projection_task %arg0, %arg2.*?"
        r"(%\w+) = ftlpu\.stream\.projection_task %arg0, %arg3.*?"
        r"(%\w+) = ftlpu\.stream\.rope_task \1.*?"
        r"(%\w+) = ftlpu\.stream\.rope_task \2.*?"
        r"(%\w+) = ftlpu\.stream\.batch_matmul_task \4, \5.*?"
        r"(%\w+) = ftlpu\.stream\.softmax_task \6.*?"
        r"(%\w+) = ftlpu\.stream\.transpose_task \7.*?"
        r"(%\w+) = ftlpu\.stream\.transpose_task \3.*?"
        r"(%\w+) = ftlpu\.stream\.batch_matmul_task \8, \9.*?"
        r"(%\w+) = ftlpu\.stream\.projection_task \10, %arg4",
        re.DOTALL,
    )
    if not graph_pattern.search(text):
        raise AssertionError("attention Stream tasks are not connected by the expected SSA DAG")

    roundtrip = args.output.with_suffix(".roundtrip.mlir")
    run(args.tool, args.output, roundtrip, "ftlpu-stablehlo-to-kernel")
    roundtrip_text = roundtrip.read_text(encoding="utf-8")
    for operation, expected in expected_counts.items():
        if roundtrip_text.count(operation) != expected:
            raise AssertionError(f"{operation} did not survive parse/print roundtrip")


if __name__ == "__main__":
    main()
