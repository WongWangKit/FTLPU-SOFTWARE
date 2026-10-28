#!/usr/bin/env python3
"""Checks that the Schedule verifier rejects physical resource overlap."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        str(args.tool), "--input", str(args.input), "--output", str(args.output),
        "--pipeline", "ftlpu-verify-schedule",
    ], capture_output=True, text=True)
    if result.returncode == 0:
        raise AssertionError("Schedule verifier accepted overlapping MEM transfers")
    if "overlaps at cycle 5" not in result.stderr and "ICU contexts overlap [5, 6)" not in result.stderr:
        raise AssertionError(f"missing overlap diagnostic: {result.stderr}")
    # Intervals arrive in graph order, not necessarily chronological order.
    # Adjacency is legal, but a command inside a strided domain's idle gap
    # still conflicts with its sole active ICU context.
    for name, domains, legal in (
        ('unordered', [(20, 3, 2), (0, 3, 2), (5, 1, 1)], True),
        ('gap_conflict', [(20, 3, 2), (0, 3, 2), (3, 1, 1)], False),
        ('prior_overlap', [(20, 3, 2), (0, 3, 2), (19, 3, 1)], False),
    ):
        operations = [f'''ftlpu.schedule.mem_transfer {{address = 0 : i64,
            address_stride = 0 : i64, cycle = {start} : i64, hemisphere = 0 : i64,
            opcode = "read", packed_stream = 0 : i64, repeat_count = {count} : i64,
            repeat_interval = {stride} : i64, slice = 0 : i64}}'''
            for start, count, stride in domains]
        fixture = args.output.parent / f'{name}.mlir'
        fixture.write_text('module { func.func @test() {\n' + '\n'.join(operations) + '\nreturn } }')
        check = subprocess.run([str(args.tool), '--input', str(fixture), '--output', str(args.output),
                                '--pipeline', 'ftlpu-verify-schedule'], capture_output=True, text=True)
        if (check.returncode == 0) != legal:
            raise AssertionError(f'{name}: {check.stderr}')


if __name__ == "__main__":
    main()
