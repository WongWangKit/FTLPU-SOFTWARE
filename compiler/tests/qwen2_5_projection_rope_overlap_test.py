#!/usr/bin/env python3
"""Check that Q projection/RoPE overlap follows the compiler switch."""

from __future__ import annotations

import argparse
from collections import Counter
import re
import subprocess
from pathlib import Path


def integer_attr(line: str, name: str, default: int | None = None) -> int:
    match = re.search(rf"(?<![\w]){re.escape(name)} = (-?\d+) : i64", line)
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise AssertionError(f"missing {name}: {line.strip()}")


def issue_interval(line: str) -> tuple[int, int]:
    """Return the half-open issue interval of a closed-form schedule domain."""
    start = integer_attr(line, "cycle")
    span = (
        (integer_attr(line, "repeat_count", 1) - 1)
        * integer_attr(line, "repeat_interval", 1)
        + (integer_attr(line, "wave_count", 1) - 1)
        * integer_attr(line, "wave_interval", 1)
        + (integer_attr(line, "group_count", 1) - 1)
        * integer_attr(line, "group_interval", 1)
    )
    return start, start + span + 1


def q_schedule_lines(schedule: Path) -> list[str]:
    lines = schedule.read_text(encoding="utf-8").splitlines()

    def first_weight_read(binding: int) -> int:
        return next(
            (index for index, line in enumerate(lines)
             if "ftlpu.schedule.mem_transfer" in line
             and 'opcode = "read"' in line
             and f"address_binding = {binding} : i64" in line),
            -1,
        )

    # The checked Qwen2.5 fixture has %arg2 = Q weight and %arg3 = K weight.
    # Attention lowering emits all Q operations before starting K emission.
    q_begin = first_weight_read(2)
    k_begin = first_weight_read(3)
    if q_begin < 0 or k_begin <= q_begin:
        raise AssertionError("cannot locate the Q/K projection boundary")
    return lines[q_begin:k_begin]


def inspect_q(schedule: Path) -> dict[str, object]:
    lines = q_schedule_lines(schedule)
    transfers = [line for line in lines
                 if "ftlpu.schedule.mem_transfer" in line]
    # MXM BF16 results enter rope_staging on streams 32/33. RoPE consumes
    # that staging area via stream 32; Q activation and weight reads use
    # different stream indices, while bias/table reads use 40/34/38.
    raw_writes = [
        issue_interval(line) for line in transfers
        if 'opcode = "write"' in line
        and integer_attr(line, "packed_stream") in (32, 33)
        and "address_binding =" not in line
    ]
    staging_reads = [
        issue_interval(line) for line in transfers
        if 'opcode = "read"' in line
        and integer_attr(line, "packed_stream") in (32, 33)
        and "address_binding =" not in line
    ]
    rope_products = [
        issue_interval(line) for line in lines
        if "ftlpu.schedule.vxm " in line
        and 'opcode = "multiply"' in line
        and "queue = 0 : i64" in line
        and "lhs_index = 32 : i64" in line
        and "rhs_index = 34 : i64" in line
    ]
    projections = [
        issue_interval(line) for line in lines
        if "ftlpu.schedule.mxm_issue" in line
        and 'opcode = "compute"' in line
    ]
    if not all((raw_writes, staging_reads, rope_products, projections)):
        raise AssertionError(
            "incomplete Q projection/RoPE schedule: "
            f"writes={len(raw_writes)}, reads={len(staging_reads)}, "
            f"products={len(rope_products)}, computes={len(projections)}"
        )
    return {
        "last_raw_write_end": max(end for _, end in raw_writes),
        "first_staging_read": min(start for start, _ in staging_reads),
        "first_product": min(start for start, _ in rope_products),
        "projections": projections,
        "rope_products": rope_products,
    }


def placement_region(stream_text: str, name: str) -> tuple[int, int]:
    placement = re.search(rf"\b{re.escape(name)} = \{{([^{{}}]*)", stream_text)
    if placement is None:
        raise AssertionError(f"missing {name} placement")
    attributes = placement.group(1)
    return (integer_attr(attributes, "base_row"),
            integer_attr(attributes, "instruction_count"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schedules: dict[str, Path] = {}
    for setting in ("off", "on"):
        stream = args.output_dir / f"qwen2_5_seq32_{setting}.stream.mlir"
        schedule = args.output_dir / f"qwen2_5_seq32_{setting}.schedule.mlir"
        common = [
            "--target-config", str(args.target_config),
            "--weight-bank", "0", "--mxm-execution", "vector",
        ]
        subprocess.run([
            str(args.opt), "--input", str(args.input), "--output",
            str(stream), "--pipeline", "ftlpu-stablehlo-to-stream",
            *(["--projection-rope-overlap", "on"] if setting == "on" else []),
            *common,
        ], check=True)
        if setting == "off":
            stream_text = stream.read_text(encoding="utf-8")
            query_base, query_rows = placement_region(stream_text, "query")
            staging_base, staging_rows = placement_region(
                stream_text, "rope_staging")
            if staging_base < query_base + query_rows or staging_rows < 1536:
                raise AssertionError(
                    "serial RoPE staging overlaps Query IW or cannot hold "
                    "all Q heads: "
                    f"query=[{query_base},{query_base + query_rows}), "
                    f"staging=[{staging_base},{staging_base + staging_rows})"
                )
        # The second compiler invocation inherits the setting from Stream IR.
        subprocess.run([
            str(args.opt), "--input", str(stream), "--output",
            str(schedule), "--pipeline", "ftlpu-stream-to-schedule",
            *common,
        ], check=True)
        schedules[setting] = schedule

    serial = inspect_q(schedules["off"])
    if not (serial["last_raw_write_end"]
            <= serial["first_staging_read"]
            <= serial["first_product"]):
        raise AssertionError(
            "overlap=off started Q RoPE before the complete Q projection "
            f"was staged: {serial}"
        )
    if any(max(mxm[0], vxm[0]) < min(mxm[1], vxm[1])
           for mxm in serial["projections"]
           for vxm in serial["rope_products"]):
        raise AssertionError("overlap=off ran Q MXM and Q RoPE concurrently")

    # Q/K/V each need one closed-form read per active physical MEM queue.
    serial_transfers = [
        line for line in schedules["off"].read_text(encoding="utf-8").splitlines()
        if "ftlpu.schedule.mem_transfer" in line
        and 'opcode = "read"' in line
    ]
    projection_boundaries: dict[int, tuple[int, int]] = {}
    for projection, binding, outer_count in (("Q", 2, 24),
                                              ("V", 4, 4),
                                              ("K", 3, 4)):
        weights = [line for line in serial_transfers
                   if integer_attr(line, "address_binding", -1) == binding]
        if not weights:
            raise AssertionError(f"serial {projection} weight reads are missing")
        projection_start = min(issue_interval(line)[0] for line in weights)
        projection_end = max(issue_interval(line)[1] for line in weights)
        projection_boundaries[binding] = (projection_start, projection_end)
        activations = [
            line for line in serial_transfers
            if "address_binding =" not in line
            and integer_attr(line, "packed_stream") in (0, 1)
            and projection_start - 2 <= integer_attr(line, "cycle") < projection_end
        ]
        for label, reads, expected_queues in (("weight", weights, 16),
                                              ("activation", activations, 4)):
            queues = Counter((integer_attr(line, "hemisphere"),
                              integer_attr(line, "slice"),
                              integer_attr(line, "bank")) for line in reads)
            if len(queues) != expected_queues or any(count != 1
                                                      for count in queues.values()):
                raise AssertionError(
                    f"serial {projection} {label} needs one READ_3D per "
                    f"physical queue: {queues}"
                )
            if any(integer_attr(line, "group_count") != outer_count
                   for line in reads):
                raise AssertionError(
                    f"serial {projection} {label} did not cover all "
                    "output halves"
                )

    # The serial Q/V/K projection owns one MXM descriptor of each kind per
    # hemisphere. Raw staging writes likewise cover the whole projection in
    # one WRITE_3D per active physical MEM queue.
    serial_lines = schedules["off"].read_text(encoding="utf-8").splitlines()
    def first_projection_weight(binding: int) -> int:
        return next(index for index, line in enumerate(serial_lines)
                    if "ftlpu.schedule.mem_transfer" in line
                    and 'opcode = "read"' in line
                    and integer_attr(line, "address_binding", -1) == binding)

    starts = {binding: first_projection_weight(binding)
              for binding in (2, 4, 3, 5)}
    for name, binding, next_binding, halves in (("Q", 2, 4, 24),
                                                ("V", 4, 3, 4),
                                                ("K", 3, 5, 4)):
        section = serial_lines[starts[binding]:starts[next_binding]]
        projection_end = projection_boundaries[binding][1]
        for opcode in ("iw", "compute"):
            domains = [line for line in section
                       if "ftlpu.schedule.mxm_issue" in line
                       and f'opcode = "{opcode}"' in line
                       and integer_attr(line, "cycle") < projection_end]
            if len(domains) != 2 or any(
                    integer_attr(line, "group_count") != halves
                    for line in domains):
                raise AssertionError(
                    f"serial {name} MXM {opcode} needs one full projection "
                    f"domain per hemisphere: {len(domains)}")
        dequants = [line for line in section
                    if "ftlpu.schedule.mxm_dequant" in line
                    and integer_attr(line, "cycle") < projection_end]
        if len(dequants) != 2 or any(
                integer_attr(line, "group_count") != halves
                for line in dequants):
            raise AssertionError(
                f"serial {name} MXM dequant needs one full projection "
                f"domain per hemisphere: {len(dequants)}")
        writes = [line for line in section
                  if "ftlpu.schedule.mem_transfer" in line
                  and 'opcode = "write"' in line
                  and integer_attr(line, "group_count", 1)
                      == (halves if name == "V" else halves // 2)
                  and integer_attr(line, "repeat_count", 1) == 4
                  and (integer_attr(line, "outer_group_size", 1) == 2
                       if name == "V"
                       else integer_attr(line, "wave_count", 1) == 2)]
        queues = Counter((integer_attr(line, "hemisphere"),
                          integer_attr(line, "slice"),
                          integer_attr(line, "bank")) for line in writes)
        if not queues or any(count != 1 for count in queues.values()):
            raise AssertionError(
                f"serial {name} raw result needs one WRITE_3D per "
                f"physical queue: {queues}")

    overlapped = inspect_q(schedules["on"])
    first_product = overlapped["first_product"]
    if not any(
        mxm[0] > first_product
        and max(mxm[0], vxm[0]) < min(mxm[1], vxm[1])
        for mxm in overlapped["projections"]
        for vxm in overlapped["rope_products"]
    ):
        raise AssertionError(
            "overlap=on did not overlap a later Q projection with an "
            f"earlier Q RoPE product: {overlapped}"
        )
    print(
        "Q projection/RoPE switch: "
        f"off raw-write-end={serial['last_raw_write_end']} "
        f"first-read={serial['first_staging_read']} "
        f"first-product={serial['first_product']}; "
        f"on first-product={first_product}",
        flush=True,
    )


if __name__ == "__main__":
    main()
