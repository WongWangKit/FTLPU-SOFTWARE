#!/usr/bin/env python3
"""Checks Schedule IR MEM exclusivity and producer/consumer readiness."""

import argparse
import re
import subprocess
from pathlib import Path


def integer(line, name):
    match = re.search(rf"\b{name} = (\d+) : i64", line)
    if not match:
        raise AssertionError(f"missing {name}: {line}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(args.tool), "--input", str(args.input), "--output", str(args.output),
         "--pipeline", "ftlpu-stablehlo-to-schedule"],
        check=True,
    )
    lines = args.output.read_text(encoding="utf-8").splitlines()
    mem_ops = [line for line in lines if "ftlpu.schedule.mem_read" in line
               or "ftlpu.schedule.mem_write" in line]
    windows = {}
    for line in mem_ops:
        slice_id = integer(line, "slice")
        start = integer(line, "cycle")
        end = start + integer(line, "duration")
        for previous_start, previous_end in windows.setdefault(slice_id, []):
            if start < previous_end and end > previous_start:
                raise AssertionError(
                    f"overlapping MEM slice {slice_id} windows: "
                    f"[{previous_start},{previous_end}) and [{start},{end})"
                )
        windows[slice_id].append((start, end))

    activation_reads = [line for line in mem_ops if 'role = "activation"' in line]
    writes = [line for line in mem_ops if "ftlpu.schedule.mem_write" in line]
    if len(activation_reads) != 3 or len(writes) != 3:
        raise AssertionError("expected three scheduled matmuls")


if __name__ == "__main__":
    main()
