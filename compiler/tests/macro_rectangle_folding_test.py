#!/usr/bin/env python3
"""Checks that an affine interleaved MEM window becomes one 2-D Macro."""

import argparse
import re
import subprocess
from pathlib import Path


def write_grid(path: Path, distinct_streams: bool,
               precompressed_repeats: bool,
               existing_outer: bool) -> None:
    operations = []
    if existing_outer:
        for offset in range(4):
            for opcode, cycle_delta, stream in (
                ("read", 0, 0), ("write", 3, 8)
            ):
                operations.append(
                    "    ftlpu.command.mem {"
                    f"cycle = {offset * 8 + cycle_delta} : i64, "
                    f"queue = 0 : i64, opcode = \"{opcode}\", "
                    f"address = {offset} : i64, "
                    f"packed_stream = {stream} : i64, "
                    "repeat_count = 1 : i64, repeat_interval = 1 : i64, "
                    "address_stride = 0 : i64, wave_count = 16 : i64, "
                    "wave_interval = 400 : i64, "
                    "wave_address_stride = 4 : i64}"
                )
    else:
        window = 4 if precompressed_repeats else 32
        repeat_count = 8 if precompressed_repeats else 1
        repeat_interval = 4 if precompressed_repeats else 1
        address_stride = 64 if precompressed_repeats else 0
        for outer in range(4):
            for offset in range(window):
                cycle = outer * 32 + offset
                address = offset * 16
                stream = offset if distinct_streams else 0
                operations.append(
                    "    ftlpu.command.mem {"
                    f"cycle = {cycle} : i64, queue = 0 : i64, "
                    "opcode = \"read\", "
                    f"address = {address} : i64, "
                    f"packed_stream = {stream} : i64, "
                    f"repeat_count = {repeat_count} : i64, "
                    f"repeat_interval = {repeat_interval} : i64, "
                    f"address_stride = {address_stride} : i64}}"
                )
    path.write_text(
        "module {\n  func.func @main() {\n"
        + "\n".join(operations)
        + "\n    return\n  }\n}\n",
        encoding="utf-8",
    )


def run_case(args: argparse.Namespace, name: str, distinct_streams: bool,
             expected_macros: int, expected_peak: int,
             precompressed_repeats: bool = False,
             existing_outer: bool = False) -> None:
    source = args.output_dir / f"{name}.mlir"
    baseline = args.output_dir / f"{name}-none.ftlpu"
    macro = args.output_dir / f"{name}-macro.ftlpu"
    write_grid(source, distinct_streams, precompressed_repeats,
               existing_outer)

    common = [str(args.translate), "--input", str(source)]
    subprocess.run(
        common + ["--output", str(baseline), "--icu-compression", "none"],
        check=True,
    )
    subprocess.run(
        common + ["--output", str(macro), "--icu-compression", "macro",
                  "--verify-icu-issues"],
        check=True,
    )

    inspection = subprocess.run(
        [str(args.inspect), str(macro), "--compare", str(baseline)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if "binary compare result=equivalent" not in inspection:
        raise RuntimeError(f"{name}: Macro changed the logical issue stream")
    aggregate = next(
        (line for line in inspection.splitlines()
         if line.startswith("binary aggregate ")),
        "",
    )
    match = re.search(r"\bmacro=(\d+)\b", aggregate)
    if not match or int(match.group(1)) != expected_macros:
        raise RuntimeError(
            f"{name}: expected {expected_macros} Macro descriptors, got: "
            f"{aggregate or '<no aggregate>'}"
        )
    if "macro_expanded=128" not in aggregate:
        raise RuntimeError(f"{name}: Macro did not preserve all 128 issues")

    execution = subprocess.run(
        [str(args.runtime_test), str(baseline), str(macro)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if f"peak_macro_contexts={expected_peak}" not in execution:
        raise RuntimeError(
            f"{name}: expected peak context count {expected_peak}, got: "
            f"{execution.strip()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--inspect", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The exact 32 x 4 pattern discussed in the Macro context review folds to
    # one descriptor when stream/opcode and all other instruction fields match.
    run_case(args, "affine-grid", False, expected_macros=1, expected_peak=1)

    # Four interleaved columns, each already represented by an eight-element
    # Repeat, are the same 32-wide affine inner row and must also fold.
    run_case(args, "affine-repeat-grid", False,
             expected_macros=1, expected_peak=1,
             precompressed_repeats=True)

    # Schedule lowering may already occupy the outer dimension. Alternating
    # Read/Write columns can still be merged into the unused inner dimension.
    run_case(args, "existing-outer-grid", False,
             expected_macros=2, expected_peak=2,
             existing_outer=True)

    # Stream is not an induction field in Macro v1. Changing it across the
    # inner window must retain one outer Macro per stream without miscompiling.
    run_case(args, "distinct-stream-grid", True,
             expected_macros=32, expected_peak=32)


if __name__ == "__main__":
    main()
