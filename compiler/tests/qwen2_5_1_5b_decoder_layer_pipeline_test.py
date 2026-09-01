#!/usr/bin/env python3
"""Builds the heavyweight Qwen2.5-1.5B decoder-layer executable pipeline."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], phase: str) -> None:
    print(f"[{phase}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def integer_attr(line: str, name: str, default: int | None = None) -> int:
    match = re.search(rf"{name} = (\d+) : i64", line)
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise AssertionError(f"missing {name} in schedule operation: {line.strip()}")


def repeated_intervals(line: str) -> list[tuple[int, int]]:
    cycle = integer_attr(line, "cycle")
    repeat_count = integer_attr(line, "repeat_count", 1)
    repeat_interval = integer_attr(line, "repeat_interval", 1)
    wave_count = integer_attr(line, "wave_count", 1)
    wave_interval = integer_attr(line, "wave_interval", 0)
    duration = (repeat_count - 1) * repeat_interval + 1
    return [
        (cycle + wave * wave_interval,
         cycle + wave * wave_interval + duration)
        for wave in range(wave_count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--weight-bank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stablehlo = args.output_dir / "decoder_layer.stablehlo.mlir"
    stream = args.output_dir / "decoder_layer.stream.mlir"
    schedule = args.output_dir / "decoder_layer.schedule.mlir"
    stale_command = args.output_dir / "decoder_layer.command.mlir"
    binary = args.output_dir / "decoder_layer.ftlpu"
    stale_command.unlink(missing_ok=True)
    shutil.copyfile(args.input, stablehlo)

    common = [
        "--mxm-execution", "vector", "--ffn-schedule", "tail",
        "--target-config", str(args.target_config),
        "--weight-bank", str(args.weight_bank),
        "--rmsnorm-strategy", "vxm-feedback",
        "--icu-macro-schedule",
    ]
    run([
        str(args.opt), "--input", str(stablehlo), "--output", str(stream),
        "--pipeline", "ftlpu-stablehlo-to-stream", *common,
    ], "stablehlo-to-stream")
    run([
        str(args.opt), "--input", str(stream), "--output", str(schedule),
        "--pipeline", "ftlpu-stream-to-compressed-schedule", *common,
    ], "stream-to-schedule")

    schedule_markers = {
        "ftlpu.schedule.compressed", 'name = "qkv"',
        'name = "softmax"', 'name = "o_proj"',
        'name = "rmsnorm.feedback"', 'name = "ffn.down.vector"',
        'accumulator_destination = "stream"',
        'accumulator_clear = true',
    }
    qkv_interval: tuple[int, int] | None = None
    rope_product_intervals: list[tuple[int, int]] = []
    mxm_compute_intervals: list[tuple[int, int]] = []
    with schedule.open(encoding="utf-8") as source:
        for line in source:
            schedule_markers = {
                marker for marker in schedule_markers if marker not in line
            }
            if "ftlpu.schedule.vxm" in line:
                queue = re.search(r"queue = (\d+) : i64", line)
                if queue and int(queue.group(1)) >= 8:
                    raise AssertionError(
                        "Schedule IR addresses a physical VXM ALU instead of "
                        f"one of the 8 mirrored control queues: {line.strip()}"
                    )
                if all(marker in line for marker in (
                    'queue = 0 : i64', 'opcode = "multiply"',
                    'lhs_index = 32 : i64', 'rhs_index = 34 : i64',
                    'repeat_count = 32 : i64',
                )):
                    rope_product_intervals.extend(repeated_intervals(line))
            if ('ftlpu.schedule.timeline' in line
                    and 'name = "qkv"' in line):
                qkv_interval = (
                    integer_attr(line, "start"), integer_attr(line, "end")
                )
            if "ftlpu.schedule.sxm" in line:
                partial = tuple(attribute for attribute in
                                ("input_row", "output_row", "output_tile")
                                if attribute in line)
                if partial:
                    raise AssertionError(
                        "SXM must operate across every tile; partial attributes "
                        f"{partial} are illegal: {line.strip()}"
                    )
                for field in ("source_streams", "destination_streams"):
                    match = re.search(rf"{field} = \[([^\]]+)\]", line)
                    if not match or len(re.findall(r"\d+", match.group(1))) != 16:
                        raise AssertionError(
                            f"SXM {field} is not full-width: {line.strip()}"
                        )
            if "ftlpu.schedule.mxm_issue" not in line:
                continue
            if 'opcode = "compute"' in line:
                mxm_compute_intervals.extend(repeated_intervals(line))
            match = re.search(r"accumulator_address = (\d+) : i64", line)
            if not match:
                continue
            address = int(match.group(1))
            limit = 8192
            if address >= limit:
                raise AssertionError(
                    f"MXM accumulator address {address} exceeds {limit} rows"
                )
    if schedule_markers:
        raise AssertionError(
            f"Schedule IR is missing {sorted(schedule_markers)}"
        )
    if qkv_interval is None:
        raise AssertionError("Schedule IR is missing the QKV timeline")
    qkv_mxm_intervals = [
        interval for interval in mxm_compute_intervals
        if interval[0] < qkv_interval[1] and interval[1] > qkv_interval[0]
    ]
    overlaps = [
        (rope, mxm)
        for rope in rope_product_intervals
        for mxm in qkv_mxm_intervals
        if max(rope[0], mxm[0]) < min(rope[1], mxm[1])
    ]
    if not overlaps:
        raise AssertionError(
            "QKV schedule does not overlap any VXM RoPE product window with "
            "an MXM projection compute window"
        )
    first_rope, first_mxm = overlaps[0]
    print(
        "QKV/RoPE overlap: "
        f"{len({rope for rope, _ in overlaps})} RoPE windows; "
        f"first RoPE {first_rope} with MXM {first_mxm}",
        flush=True,
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
        raise AssertionError("Qwen decoder-layer binary is unexpectedly small")
    print(
        f"Qwen2.5-1.5B decoder-layer executable: {binary} "
        f"({binary.stat().st_size} bytes)", flush=True,
    )


if __name__ == "__main__":
    main()
