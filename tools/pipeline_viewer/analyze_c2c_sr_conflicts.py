#!/usr/bin/env python3
"""Estimate legacy C2C conflicts with compute stream-register usage."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def load_windows(path: Path, resource: str,
                 full_trace: bool) -> list[tuple[int, int, str]]:
    windows = []
    first = None
    last = None
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            start = int(row["start"])
            end = int(row["end"])
            first = start if first is None else min(first, start)
            last = end if last is None else max(last, end)
            if row["resource"] == resource:
                windows.append((start, end, row["detail"]))
    if full_trace and first is not None and last is not None:
        return [(first, last, "full trace")]
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--hemisphere", choices=("E", "W"), default="E")
    parser.add_argument(
        "--stream-direction", choices=("E", "W"), default="W")
    parser.add_argument("--stream-count", type=int, default=32)
    parser.add_argument("--c2c-width", type=int, default=8)
    parser.add_argument("--mxm-result-width", type=int, default=4)
    parser.add_argument("--full-trace", action="store_true",
                        help="model one continuous C2C transfer")
    args = parser.parse_args()

    c2c_resource = f"C2C.{args.hemisphere}.Prefetch"
    windows = load_windows(args.trace, c2c_resource, args.full_trace)
    if not windows:
        raise SystemExit(f"trace has no {c2c_resource} windows")
    first_cycle = min(start for start, _, _ in windows)
    last_cycle = max(end for _, end, _ in windows)
    occupancy = [bytearray(last_cycle - first_cycle)
                 for _ in range(args.stream_count)]
    resources = [set() for _ in range(args.stream_count)]
    stream_pattern = re.compile(
        rf"\b{args.stream_direction}(\d+)\b")
    mxm_output_pattern = re.compile(
        rf"\bout={args.stream_direction}(\d+)\b")

    with args.trace.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["resource"].startswith("C2C."):
                continue
            resource_parts = row["resource"].split(".")
            family = resource_parts[0]
            if (family in {"MEM", "MXM", "SXM", "ACC"}
                    and (len(resource_parts) < 2
                         or not resource_parts[1].startswith(
                             args.hemisphere))):
                continue
            start = max(first_cycle, int(row["start"]))
            end = min(last_cycle, int(row["end"]))
            if end <= start:
                continue
            streams = {
                int(value) for value in stream_pattern.findall(row["detail"])
                if int(value) < args.stream_count
            }
            if (row["resource"].startswith("MXM.")
                    and row["resource"].endswith(".Compute")):
                match = mxm_output_pattern.search(row["detail"])
                if match:
                    base = int(match.group(1))
                    streams.update(range(
                        base, min(args.stream_count,
                                  base + args.mxm_result_width)))
            for stream in streams:
                occupancy[stream][start - first_cycle:end - first_cycle] = (
                    b"\1" * (end - start))
                resources[stream].add(row["resource"])

    print(f"resource={c2c_resource} windows={len(windows)} "
          f"cycle_range={first_cycle}..{last_cycle}")
    for base in range(0, args.stream_count, args.c2c_width):
        group = range(base, min(args.stream_count, base + args.c2c_width))
        collisions = []
        for start, end, _ in windows:
            collisions.append(sum(
                any(occupancy[stream][cycle - first_cycle]
                    for stream in group)
                for cycle in range(start, end)))
        print(
            f"{args.stream_direction}{base}.."
            f"{args.stream_direction}{group.stop - 1}: "
            f"collision_cycles={sum(collisions)} "
            f"windows_with_conflict={sum(value > 0 for value in collisions)}"
            f"/{len(windows)} max_window={max(collisions)} "
            f"per_window={collisions}")
    for stream, active in enumerate(occupancy):
        cycles = sum(active)
        if cycles:
            print(f"{args.stream_direction}{stream}: active_cycles={cycles} "
                  f"resources={','.join(sorted(resources[stream]))}")

    print("note=conservative whole-stream check; CSV does not encode "
          "per-column SR cells or implicit VXM input streams")


if __name__ == "__main__":
    main()
