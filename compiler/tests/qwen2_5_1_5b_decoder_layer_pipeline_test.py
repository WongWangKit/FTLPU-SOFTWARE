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
    match = re.search(rf"{name} = (-?\d+) : i64", line)
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
    wave_interval = integer_attr(line, "wave_interval", 1)
    group_count = integer_attr(line, "group_count", 1)
    group_interval = integer_attr(line, "group_interval", 1)
    duration = (repeat_count - 1) * repeat_interval + 1
    return [
        (cycle + group * group_interval + wave * wave_interval,
         cycle + group * group_interval + wave * wave_interval + duration)
        for group in range(group_count)
        for wave in range(wave_count)
    ]


def coarse_interval(line: str) -> tuple[int, int]:
    """Returns the full lifetime of one non-preemptible ICU instruction."""
    cycle = integer_attr(line, "cycle")
    repeat_count = integer_attr(line, "repeat_count", 1)
    repeat_interval = integer_attr(line, "repeat_interval", 1)
    wave_count = integer_attr(line, "wave_count", 1)
    wave_interval = integer_attr(line, "wave_interval", 1)
    group_count = integer_attr(line, "group_count", 1)
    group_interval = integer_attr(line, "group_interval", 1)
    last_issue = (
        cycle
        + (repeat_count - 1) * repeat_interval
        + (wave_count - 1) * wave_interval
        + (group_count - 1) * group_interval
    )
    return cycle, last_issue + 1


def flat_memory_plan_placement(
        stream_text: str, name: str) -> tuple[int, int, frozenset[int]]:
    """Returns bank/base/slices for a flat attention memory-plan entry."""
    matches = re.findall(rf"\b{re.escape(name)} = \{{([^{{}}]+)\}}",
                         stream_text)
    if not matches:
        raise AssertionError(
            f"attention memory plan is missing {name} placement"
        )
    placements: set[tuple[int, int, frozenset[int]]] = set()
    for fields in matches:
        slices_match = re.search(r"slices = \[([^]]+)\]", fields)
        if not slices_match:
            raise AssertionError(
                f"attention {name} placement has no physical slices: "
                f"{fields}"
            )
        placements.add((
            integer_attr(fields, "bank"),
            integer_attr(fields, "base_row"),
            frozenset(int(value) for value in
                      re.findall(r"\d+", slices_match.group(1))),
        ))
    if len(placements) != 1:
        raise AssertionError(
            f"attention {name} placement is inconsistent: {placements}"
        )
    return placements.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--weight-bank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--ffn-schedule", choices=("tail", "fused"),
                        default="fused")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kv-cache-capacity", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stablehlo = args.output_dir / "decoder_layer.stablehlo.mlir"
    stream = args.output_dir / "decoder_layer.stream.mlir"
    schedule = args.output_dir / "decoder_layer.schedule.mlir"
    command = args.output_dir / "decoder_layer.command.mlir"
    binary = args.output_dir / "decoder_layer.ftlpu"
    shutil.copyfile(args.input, stablehlo)

    common = [
        "--mxm-execution", "vector", "--ffn-schedule", args.ffn_schedule,
        "--projection-rope-overlap", "on",
        "--target-config", str(args.target_config),
        "--weight-bank", str(args.weight_bank),
    ]
    if args.kv_cache_capacity:
        common += ["--kv-cache-capacity", str(args.kv_cache_capacity)]
    run([
        str(args.opt), "--input", str(stablehlo), "--output", str(stream),
        "--pipeline", "ftlpu-stablehlo-to-stream", *common,
    ], "stablehlo-to-stream")
    run([
        str(args.opt), "--input", str(stream), "--output", str(schedule),
        "--pipeline", "ftlpu-stream-to-schedule", *common,
    ], "stream-to-schedule")

    stream_text = stream.read_text(encoding="utf-8")
    staging_bank, staging_base, staging_slices = (
        flat_memory_plan_placement(stream_text, "input_staging")
    )
    pong_bank, pong_base, pong_slices = flat_memory_plan_placement(
        stream_text, "input_staging_pong"
    )
    rope_staging_bank, rope_staging_base, rope_staging_slices = (
        flat_memory_plan_placement(stream_text, "rope_staging")
    )
    staging_mirror_bank, staging_mirror_base, staging_mirror_slices = (
        flat_memory_plan_placement(stream_text, "rope_staging_mirror")
    )
    mirror_spare_slices = staging_mirror_slices - rope_staging_slices
    bias_bank, _, bias_slices = flat_memory_plan_placement(
        stream_text, "query_bias"
    )
    mirror_bank, _, mirror_slices = flat_memory_plan_placement(
        stream_text, "rope_mirror"
    )
    if ((bias_bank, bias_slices) != (0, frozenset(range(42, 46)))
            or (mirror_bank, mirror_slices) !=
            (0, frozenset(range(46, 50)))
            or (staging_mirror_bank, mirror_spare_slices) !=
            (0, frozenset({28, 29, 30, 32}))
            or len(staging_mirror_slices) != 16
            or staging_mirror_base < 384
            or staging_mirror_base + 4 * 32 > staging_base):
        raise AssertionError(
            "Q bias, mirrored RoPE table and staging remap must avoid "
            "pre-execution weight-page slices and each other: "
            f"bias={(bias_bank, sorted(bias_slices))}, "
            f"table={(mirror_bank, sorted(mirror_slices))}, "
            f"staging_mirror={(staging_mirror_bank, staging_mirror_base, sorted(staging_mirror_slices))}"
        )
    preloaded_bank0_slices = set(range(20, 28)) | {
        31, 33, 34, 35, 37, 38, 39, 41,
    }
    if not (bias_slices | mirror_slices | mirror_spare_slices).isdisjoint(
            preloaded_bank0_slices):
        raise AssertionError("Q constants or staging overlap a preloaded page")
    if (len(staging_slices) != 2 or len(pong_slices) != 2
            or not staging_slices.isdisjoint(pong_slices)
            or staging_bank != pong_bank or staging_base != pong_base):
        raise AssertionError(
            "attention activation ping-pong requires two disjoint, colocated "
            "MEM slice pairs: "
            f"primary=(bank={staging_bank}, base={staging_base}, "
            f"slices={sorted(staging_slices)}), "
            f"pong=(bank={pong_bank}, base={pong_base}, "
            f"slices={sorted(pong_slices)})"
        )
    if any(slice_id >= 20 for slice_id in staging_slices | pong_slices):
        raise AssertionError(
            "attention activation ping-pong escaped into dedicated weight "
            "slices: "
            f"primary={sorted(staging_slices)}, pong={sorted(pong_slices)}"
        )

    schedule_markers = {
        "ftlpu.schedule.closed_form", 'name = "qkv"',
        'name = "softmax"', 'name = "o_proj"',
        'name = "rmsnorm.feedback"', 'name = "ffn.down.vector"',
        'accumulator_destination = "stream"',
        'accumulator_clear = true',
    }
    qkv_interval: tuple[int, int] | None = None
    rope_interval: tuple[int, int] | None = None
    pv_interval: tuple[int, int] | None = None
    o_proj_interval: tuple[int, int] | None = None
    rmsnorm_restore_ends: list[int] = []
    ffn_weight_first_cycles: dict[int, int | None] = {10: None, 11: None}
    ffn_projection_weight_reads: dict[int, int] = {10: 0, 11: 0}
    ffn_projection_blocked_reads: dict[int, int] = {10: 0, 11: 0}
    ffn_projection_hazard_waves: dict[int, dict[int, int]] = {
        10: {}, 11: {},
    }
    qkv_weight_reads: dict[int, int] = {2: 0, 3: 0, 4: 0}
    qkv_weight_workloads: dict[int, int] = {2: 0, 3: 0, 4: 0}
    qkv_weight_domains: dict[int, dict[tuple[int, int], int]] = {
        2: {}, 3: {}, 4: {},
    }
    o_projection_weight_reads = 0
    o_projection_weight_workload = 0
    o_projection_weight_domains: dict[tuple[int, int], int] = {}
    swish_output_workloads: dict[str, int] = {}
    swish_write_workloads: dict[str, dict[tuple[str, int], int]] = {}
    fused_hidden_locations: list[tuple[int, int, frozenset[int]]] = []
    ffn_input_allocation: tuple[str, int, int, int, frozenset[int]] | None = None
    rope_product_intervals: list[tuple[int, int]] = []
    rope_combine_intervals: list[tuple[int, int]] = []
    qk_bias_low_fma = False
    qk_bias_high_fma = False
    value_bias_add = False
    bias_bindings: set[str] = set()
    binding_ready_cycles: dict[str, int] = {}
    state_bindings: dict[str, tuple[str, int]] = {}
    internal_address_bindings: set[int] = set()
    mxm_compute_intervals: list[tuple[int, int]] = []
    projection_compute_domains: dict[int, list[tuple[int, int, int, int]]] = {}
    qkv_overlap_compute_domains: list[dict[str, int | bool]] = []
    streaming_bf16_compute_intervals: list[tuple[int, int]] = []
    accumulator_read_intervals: list[tuple[int, int]] = []
    overlap_activation_reads: dict[
        tuple[int, int, int], list[tuple[int, int]]
    ] = {}
    staging_writes: dict[
        tuple[int, int, int], list[tuple[int, int]]
    ] = {}
    q_write_read_domains: dict[
        tuple[int, int, int], list[tuple[int, int, int]]
    ] = {}
    q_copy_streams: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    q_staging_accesses: list[tuple[int, int, int, int, int, str]] = []
    host_preloaded_allocations: list[
        tuple[str, int, int, int, frozenset[int]]
    ] = []
    initialized_allocations: list[
        tuple[str, int, int, int, frozenset[int]]
    ] = []
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
                if all(marker in line for marker in (
                    'queue = 0 : i64', 'opcode = "subtract"',
                    'lhs_index = 32 : i64', 'rhs_index = 34 : i64',
                    'repeat_count = 32 : i64',
                )):
                    rope_combine_intervals.extend(repeated_intervals(line))
                if all(marker in line for marker in (
                    'queue = 1 : i64', 'opcode = "fma"',
                    'lhs_index = 40 : i64', 'rhs_index = 34 : i64',
                    'output_stream = 0 : i64',
                )):
                    qk_bias_low_fma = True
                if all(marker in line for marker in (
                    'queue = 3 : i64', 'opcode = "fma"',
                    'lhs_index = 42 : i64', 'rhs_index = 38 : i64',
                    'output_stream = 2 : i64',
                )):
                    qk_bias_high_fma = True
                if all(marker in line for marker in (
                    '%arg8' , 'opcode = "add"',
                    'lhs_index = 32 : i64', 'rhs_index = 40 : i64',
                    'repeat_count = 32 : i64',
                )):
                    value_bias_add = True
                if all(marker in line for marker in (
                    'queue = 7 : i64', 'opcode = "bypass"',
                    'cast_target = "bf16"', 'output_stream = 6 : i64',
                )):
                    result = re.match(r"\s*(%\d+)\s*=", line)
                    if result:
                        swish_output_workloads[result.group(1)] = integer_attr(
                            line, "repeat_count", 1
                        )
            if "ftlpu.schedule.mem_write" in line:
                source = re.search(r"ftlpu\.schedule\.mem_write\s+(%\d+)", line)
                if source and source.group(1) in swish_output_workloads:
                    output = source.group(1)
                    hemisphere = re.search(
                        r'placement = \{.*?hemisphere = "(east|west)"', line
                    )
                    if hemisphere:
                        stream_base = integer_attr(line, "stream_base")
                        stream_count = integer_attr(line, "stream_count", 1)
                        write_workload = (
                            integer_attr(line, "repeat_count", 1)
                            * integer_attr(line, "wave_count", 1)
                            * integer_attr(line, "group_count", 1)
                        )
                        workloads = swish_write_workloads.setdefault(output, {})
                        for stream in range(stream_base,
                                            stream_base + stream_count):
                            route = (hemisphere.group(1), stream)
                            workloads[route] = (workloads.get(route, 0)
                                                + write_workload)
                    placement = re.search(r"placement = \{([^}]+)\}", line)
                    if placement:
                        fields = placement.group(1)
                        bank = re.search(r"bank = (\d+) : i64", fields)
                        base = re.search(r"base_row = (\d+) : i64", fields)
                        slices = re.search(r"slices = \[([^]]+)\]", fields)
                        if bank and base and slices:
                            fused_hidden_locations.append((
                                int(bank.group(1)), int(base.group(1)),
                                frozenset(int(value) for value in
                                          re.findall(r"\d+", slices.group(1))),
                            ))
            if "ftlpu.schedule.binding" in line:
                if 'role = "bias"' in line:
                    name = re.search(r'name = "([^"]+)"', line)
                    if name:
                        bias_bindings.add(name.group(1))
                name = re.search(r'name = "([^"]+)"', line)
                ready = re.search(r'ready_cycle = (\d+) : i64', line)
                if name and ready:
                    binding_ready_cycles[name.group(1)] = int(
                        ready.group(1)
                    )
                state_role = re.search(r'role = "(state\.kv\.[^"]+)"', line)
                if name and state_role:
                    state_bindings[name.group(1)] = (
                        state_role.group(1), integer_attr(line, "index")
                    )
                slices_match = re.search(r"slices = \[([^\]]+)\]", line)
                if slices_match:
                    name_match = re.search(r'name = "([^"]+)"', line)
                    allocation = (
                        name_match.group(1) if name_match else line.split("=")[0].strip(),
                        integer_attr(line, "bank"),
                        integer_attr(line, "base_row"),
                        integer_attr(line, "instruction_count"),
                        frozenset(int(value) for value in
                                  re.findall(r"\d+", slices_match.group(1))),
                    )
                    if allocation[0] == "rmsnorm.result.1":
                        ffn_input_allocation = allocation
                    if ('access = "input"' in line
                            and "paged_weight = true" not in line):
                        host_preloaded_allocations.append(allocation)
                    if ('access = "internal"' in line
                            and 'initializer = "none"' not in line
                            and "initializer =" in line):
                        initialized_allocations.append(allocation)
            if ('ftlpu.schedule.mem_transfer' in line
                    and 'address_binding_access = "internal"' in line):
                internal_address_bindings.add(
                    integer_attr(line, "address_binding")
                )
            if "ftlpu.schedule.mem_transfer" in line:
                bank = integer_attr(line, "bank")
                slice_id = integer_attr(line, "slice")
                hemisphere = integer_attr(line, "hemisphere")
                address = integer_attr(line, "address")
                queue = (hemisphere, slice_id, bank)
                if (slice_id in rope_staging_slices | mirror_spare_slices
                        and ((rope_staging_base <= address <
                              rope_staging_base + 4 * 32)
                             or (staging_mirror_base <= address <
                                 staging_mirror_base + 4 * 32))):
                    if (bank == staging_mirror_bank
                            and staging_mirror_base <= address <
                            staging_mirror_base + 4 * 32
                            and 'opcode = "write"' in line
                            and not 0 <= integer_attr(
                                line, "packed_stream") <= 23):
                        raise AssertionError(
                            "Q mirror write used a Q/V weight SR stream: "
                            f"{line.strip()}"
                        )
                    q_staging_accesses.append((
                        integer_attr(line, "cycle"), hemisphere, slice_id,
                        bank, address,
                        re.search(r'opcode = "([^"]+)"', line).group(1),
                    ))
                is_primary_queue = (
                    bank == staging_bank and slice_id in staging_slices
                )
                is_pong_queue = (
                    bank == pong_bank and slice_id in pong_slices
                )
                if is_primary_queue or is_pong_queue:
                    if ('opcode = "read"' in line
                            and integer_attr(line, "packed_stream") in (4, 5)):
                        overlap_activation_reads.setdefault(queue, []).append(
                            coarse_interval(line)
                        )
                    if 'opcode = "write"' in line:
                        staging_writes.setdefault(queue, []).append(
                            coarse_interval(line)
                        )
            if "ftlpu.schedule.mem_write_read_2d" in line:
                queue = (integer_attr(line, "hemisphere"),
                         integer_attr(line, "slice"),
                         integer_attr(line, "bank"))
                count0 = integer_attr(line, "count0")
                count1 = integer_attr(line, "count1")
                write_stride0 = integer_attr(line, "write_cycle_stride0")
                write_stride1 = integer_attr(line, "write_cycle_stride1")
                read_stride0 = integer_attr(line, "read_cycle_stride0")
                read_stride1 = integer_attr(line, "read_cycle_stride1")
                read_offset = integer_attr(line, "read_start_offset")
                if ((count0, count1, write_stride0, read_stride0,
                     integer_attr(line, "address_stride0")) !=
                        (4, 2, 8, 1, 1)):
                    raise AssertionError(
                        "Q raw WRITE/copy READ lost its two-half four-row "
                        f"direct domain: {line.strip()}"
                    )
                if integer_attr(line, "read_stream_outer_stride") != 0:
                    raise AssertionError(
                        "Q copy READ must keep one stream per physical "
                        f"source slice across both halves: {line.strip()}"
                    )
                read_stream_base = integer_attr(line, "read_stream_base")
                if read_stream_base not in (32 + queue[1], 40 + queue[1]):
                    raise AssertionError(
                        "Q copy READ must keep a source-slice channel in "
                        "either the Q or Q/V boundary stream partition: "
                        f"{line.strip()}"
                    )
                write_beats = {
                    i * write_stride0 + j * write_stride1
                    for j in range(count1) for i in range(count0)
                }
                read_beats = {
                    read_offset + i * read_stride0 + j * read_stride1
                    for j in range(count1) for i in range(count0)
                }
                if (len(write_beats) != 8 or len(read_beats) != 8
                        or not write_beats.isdisjoint(read_beats)):
                    raise AssertionError(
                        "Q WRITE_READ_2D has a same-bank FU issue collision: "
                        f"{line.strip()}"
                    )
                if any(read_offset + i * read_stride0 + j * read_stride1
                       <= i * write_stride0 + j * write_stride1
                       for j in range(count1) for i in range(count0)):
                    raise AssertionError(
                        "Q WRITE_READ_2D reads before its matching write: "
                        f"{line.strip()}"
                    )
                start = integer_attr(line, "cycle")
                end = start + max(
                    max(write_beats), max(read_beats)
                ) + 1
                q_write_read_domains.setdefault(queue, []).append((
                    start, end, 16,
                ))
                q_copy_streams.setdefault(queue, []).append((
                    start, read_stream_base,
                ))
            if ('ftlpu.schedule.mem_transfer' in line
                    and 'opcode = "read"' in line):
                binding_match = re.search(
                    r"address_binding = (\d+) : i64", line
                )
                if binding_match:
                    binding = int(binding_match.group(1))
                    if binding in qkv_weight_reads:
                        repeat_count = integer_attr(line, "repeat_count", 1)
                        wave_count = integer_attr(line, "wave_count", 1)
                        group_count = integer_attr(line, "group_count", 1)
                        qkv_weight_reads[binding] += 1
                        qkv_weight_workloads[binding] += (
                            repeat_count * wave_count * group_count
                        )
                        shape = (wave_count, group_count)
                        domains = qkv_weight_domains[binding]
                        domains[shape] = domains.get(shape, 0) + 1
                        if binding in (2, 3, 4) and (
                                integer_attr(line, "outer_group_size", 1) != 2
                                or integer_attr(line, "outer_inner_stride") != 4
                                or integer_attr(line, "outer_group_stride") != 384):
                            raise AssertionError(
                                "QKV weight READ_3D lost its blocked outer "
                                f"output-group address: {line.strip()}"
                            )
                    if binding == 5:
                        repeat_count = integer_attr(line, "repeat_count", 1)
                        wave_count = integer_attr(line, "wave_count", 1)
                        group_count = integer_attr(line, "group_count", 1)
                        o_projection_weight_reads += 1
                        o_projection_weight_workload += (
                            repeat_count * wave_count * group_count
                        )
                        shape = (wave_count, group_count)
                        o_projection_weight_domains[shape] = (
                            o_projection_weight_domains.get(shape, 0) + 1
                        )
            if ('ftlpu.schedule.timeline' in line
                    and 'name = "qkv"' in line):
                qkv_interval = (
                    integer_attr(line, "start"), integer_attr(line, "end")
                )
            if ('ftlpu.schedule.timeline' in line
                    and 'name = "rope"' in line):
                rope_interval = (
                    integer_attr(line, "start"), integer_attr(line, "end")
                )
            if ('ftlpu.schedule.timeline' in line
                    and 'name = "pv"' in line):
                pv_interval = (
                    integer_attr(line, "start"), integer_attr(line, "end")
                )
            if ('ftlpu.schedule.timeline' in line
                    and 'name = "o_proj"' in line):
                o_proj_interval = (
                    integer_attr(line, "start"), integer_attr(line, "end")
                )
            if ('ftlpu.schedule.timeline' in line
                    and 'name = "rmsnorm.restore_layout"' in line):
                rmsnorm_restore_ends.append(integer_attr(line, "end"))
            for binding in ffn_weight_first_cycles:
                if f"ftlpu.schedule.mem_read %arg{binding} " not in line:
                    continue
                ffn_projection_weight_reads[binding] += 1
                group_count = integer_attr(line, "group_count", 1)
                if group_count > 1:
                    for field in ("outer_group_size", "outer_inner_stride",
                                  "outer_group_stride"):
                        if f"{field} =" not in line:
                            raise AssertionError(
                                "FFN projection pair domain lost its blocked "
                                f"outer address field {field}: {line.strip()}"
                            )
                    if integer_attr(line, "outer_group_size") != 2:
                        raise AssertionError(
                            "FFN projection weight address generator must use "
                            f"two-slot outer groups: {line.strip()}"
                        )
                    ffn_projection_blocked_reads[binding] += 1
                else:
                    wave_count = integer_attr(line, "wave_count", 1)
                    waves = ffn_projection_hazard_waves[binding]
                    waves[wave_count] = waves.get(wave_count, 0) + 1
                cycle = integer_attr(line, "cycle")
                first = ffn_weight_first_cycles[binding]
                if first is None or cycle < first:
                    ffn_weight_first_cycles[binding] = cycle
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
                intervals = repeated_intervals(line)
                mxm_compute_intervals.extend(intervals)
                unit_id = integer_attr(line, "unit_id")
                projection_compute_domains.setdefault(unit_id, []).append((
                    integer_attr(line, "cycle"),
                    integer_attr(line, "repeat_count", 1),
                    integer_attr(line, "group_count", 1),
                    integer_attr(line, "group_interval", 1),
                ))
                if 'activation_stream_base = 4 : i64' in line:
                    qkv_overlap_compute_domains.append({
                        "cycle": integer_attr(line, "cycle"),
                        "repeat_count": integer_attr(line, "repeat_count", 1),
                        "repeat_interval": integer_attr(
                            line, "repeat_interval", 1),
                        "wave_count": integer_attr(line, "wave_count", 1),
                        "wave_interval": integer_attr(line, "wave_interval", 1),
                        "group_count": integer_attr(line, "group_count", 1),
                        "group_interval": integer_attr(
                            line, "group_interval", 1),
                        "unit_id": integer_attr(line, "unit_id"),
                        "toggle": 'weight_buffer_mode = "toggle_dim2"' in line,
                    })
                if all(marker in line for marker in (
                    'accumulator_destination = "stream"',
                    'accumulator_output_format = "bf16"',
                    'accumulator_clear = true',
                )):
                    streaming_bf16_compute_intervals.extend(intervals)
            if 'opcode = "accumulator_read"' in line:
                accumulator_read_intervals.extend(repeated_intervals(line))
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
    for host in host_preloaded_allocations:
        for initialized in initialized_allocations:
            host_name, host_bank, host_base, host_rows, host_slices = host
            init_name, init_bank, init_base, init_rows, init_slices = initialized
            rows_overlap = (host_base < init_base + init_rows
                            and init_base < host_base + host_rows)
            if (host_bank == init_bank and rows_overlap
                    and not host_slices.isdisjoint(init_slices)):
                raise AssertionError(
                    "host-preloaded binding aliases an initialized constant: "
                    f"{host_name} and {init_name} share bank {host_bank}, "
                    f"slices {sorted(host_slices & init_slices)}, rows "
                    f"[{max(host_base, init_base)}, "
                    f"{min(host_base + host_rows, init_base + init_rows)})"
                )
    if qkv_interval is None:
        raise AssertionError("Schedule IR is missing the QKV timeline")
    expected_q_write_read_queues = {
        (hemisphere, slice_id, rope_staging_bank)
        for hemisphere in range(2)
        for slice_id in rope_staging_slices
    }
    if (len(rope_staging_slices) != 16
            or set(q_write_read_domains) != expected_q_write_read_queues
            or any(len(domains) != 12
                   for domains in q_write_read_domains.values())
            or sum(beats for domains in q_write_read_domains.values()
                   for _, _, beats in domains) != 12 * 32 * 16):
        raise AssertionError(
            "Q projection needs one 16-beat WRITE_READ_2D instruction per "
            "output group and physical rope-staging queue: "
            f"observed={{{', '.join(f'{queue}: {len(domains)}' for queue, domains in sorted(q_write_read_domains.items()))}}}"
        )
    for queue, streams in q_copy_streams.items():
        ordered = sorted(streams)
        slice_id = queue[1]
        if (len(ordered) != 12
                or any(stream != 40 + slice_id
                       for _, stream in ordered[:-1])
                or ordered[-1][1] != 32 + slice_id):
            raise AssertionError(
                "Q copy must use E/W8..23 until its final output group, "
                f"then E/W0..15 before V activation: {queue}: {ordered}"
            )
    q_last_pair_end = max(end for domains in q_write_read_domains.values()
                          for _, end, _ in domains)
    q_staging_reads = [
        (hemisphere, slice_id, bank, address)
        for cycle, hemisphere, slice_id, bank, address, opcode
        in q_staging_accesses
        if cycle < q_last_pair_end and opcode == "read"
    ]
    local_reads = 0
    mirrored_reads = 0
    for hemisphere, _, bank, address in q_staging_reads:
        # The four output blocks reuse FIFO row offsets for every Q head, so
        # the source hemisphere cannot be inferred from the address alone.
        is_mirror_address = (
            staging_mirror_base <= address < staging_mirror_base + 4 * 32
        )
        expected_bank = (staging_mirror_bank if is_mirror_address
                         else rope_staging_bank)
        if bank != expected_bank:
            raise AssertionError(
                "Q RoPE source read used the wrong local/mirror bank: "
                f"hemisphere={hemisphere}, address={address}, bank={bank}, "
                f"expected={expected_bank}"
            )
        if bank == rope_staging_bank:
            local_reads += 1
        else:
            mirrored_reads += 1
    q_mirror_writes = [
        access for access in q_staging_accesses
        if access[0] < q_last_pair_end + 32
        and access[3] == (rope_staging_bank + 1) % 2
        and access[5] == "write"
    ]
    remapped_mirror_writes = [
        access for access in q_mirror_writes
        if access[2] in mirror_spare_slices
    ]
    if (not local_reads or not mirrored_reads
            or len(q_mirror_writes) != 12 * 32 * 2
            or len(remapped_mirror_writes) != 12 * 4 * 4):
        raise AssertionError(
            "Q staging mirror did not use bank 0 writes and matching "
            "bank 0 Product A/B reads: "
            f"local_reads={local_reads}, mirrored_reads={mirrored_reads}, "
            f"mirror_writes={len(q_mirror_writes)}, "
            f"remapped={len(remapped_mirror_writes)}"
        )
    if rope_interval is None:
        raise AssertionError("Schedule IR is missing the RoPE timeline")
    if pv_interval is None:
        raise AssertionError("Schedule IR is missing the PV timeline")
    if o_proj_interval is None:
        raise AssertionError("Schedule IR is missing the O projection timeline")
    if len(rmsnorm_restore_ends) < 2:
        raise AssertionError("Schedule IR is missing the second RMSNorm restore")
    qkv_activation_reads = {
        queue: [interval for interval in intervals
                if interval[0] < qkv_interval[1]
                and interval[1] > qkv_interval[0]]
        for queue, intervals in overlap_activation_reads.items()
    }
    qkv_activation_reads = {
        queue: intervals for queue, intervals in qkv_activation_reads.items()
        if intervals
    }
    expected_primary_queues = {
        (hemisphere, slice_id, staging_bank)
        for hemisphere in range(2) for slice_id in staging_slices
    }
    expected_pong_queues = {
        (hemisphere, slice_id, pong_bank)
        for hemisphere in range(2) for slice_id in pong_slices
    }
    expected_pingpong_queues = expected_primary_queues | expected_pong_queues
    if set(qkv_activation_reads) != expected_pingpong_queues:
        raise AssertionError(
            "QKV/RoPE overlap did not route activation READs through both "
            "activation-region staging copies: "
            f"observed={sorted(qkv_activation_reads)}, "
            f"expected={sorted(expected_pingpong_queues)}"
        )
    first_overlap_read = min(
        begin for intervals in qkv_activation_reads.values()
        for begin, _ in intervals
    )
    query_iw_writes = {
        queue: [interval for interval in staging_writes.get(queue, [])
                if first_overlap_read <= interval[0] < qkv_interval[1]]
        for queue in expected_pingpong_queues
    }
    missing_query_iw_writes = {
        queue for queue, intervals in query_iw_writes.items() if not intervals
    }
    if missing_query_iw_writes:
        raise AssertionError(
            "QKV schedule does not exercise Query-IW writes on every "
            "ping-pong MEM queue: "
            f"missing={sorted(missing_query_iw_writes)}"
        )
    mem_icu_collisions = []
    for queue, reads in qkv_activation_reads.items():
        for read in reads:
            for write in query_iw_writes[queue]:
                if max(read[0], write[0]) < min(read[1], write[1]):
                    mem_icu_collisions.append((queue, read, write))
    if mem_icu_collisions:
        raise AssertionError(
            "activation READ and Query-IW WRITE overlap on one physical MEM "
            "ICU queue: "
            f"collisions={mem_icu_collisions[:8]}"
        )
    expected_qkv_weight_reads = {2: 16, 3: 16, 4: 16}
    expected_qkv_weight_workloads = {2: 73728, 3: 12288, 4: 12288}
    expected_qkv_weight_domains = {
        2: {(48, 24): 16},
        3: {(48, 4): 16},
        4: {(48, 4): 16},
    }
    if (qkv_weight_reads != expected_qkv_weight_reads
            or qkv_weight_workloads != expected_qkv_weight_workloads
            or qkv_weight_domains != expected_qkv_weight_domains):
        raise AssertionError(
            "QKV weight READ_3D domains lost the complete reduction/half "
            "merge: "
            f"reads={qkv_weight_reads}, workloads={qkv_weight_workloads}, "
            f"domains={qkv_weight_domains}"
        )
    for unit, domains in projection_compute_domains.items():
        ordered_domains = sorted(
            (domain for domain in domains
             if qkv_interval[0] <= domain[0] < qkv_interval[1]),
            key=lambda domain: domain[0],
        )
        for name, begin, count in (("Q", 0, 24), ("V", 24, 4),
                                   ("K", 28, 4)):
            current = ordered_domains[begin:begin + count]
            if (len(current) != count
                    or any(domain[1:] != (32, 48, 32)
                           for domain in current)
                    or any(right[0] != left[0] + 48 * 32
                           for left, right in zip(current, current[1:]))):
                raise AssertionError(
                    f"{name} projection has an MXM bubble between output "
                    f"halves: unit={unit}, domains={current}"
                )
    if (o_projection_weight_reads != 16
            or o_projection_weight_workload != 73_728
            or o_projection_weight_domains != {(48, 24): 16}):
        raise AssertionError(
            "O-projection weight reads were not emitted as one maximal 3-D "
            "domain per physical MEM queue: "
            f"reads={o_projection_weight_reads}, "
            f"workload={o_projection_weight_workload}, "
            f"domains={o_projection_weight_domains}"
        )
    gate_first = ffn_weight_first_cycles[10]
    up_first = ffn_weight_first_cycles[11]
    if gate_first is None or up_first is None:
        raise AssertionError("Schedule IR is missing Gate/Up paged weight reads")
    expected_projection_reads = {10: 96, 11: 96}
    if ffn_projection_weight_reads != expected_projection_reads:
        raise AssertionError(
            "Gate/Up weight reads were not emitted as maximal closed 3-D "
            f"domains: observed={ffn_projection_weight_reads}, "
            f"expected={expected_projection_reads}"
        )
    expected_blocked_reads = {10: 64, 11: 64}
    if ffn_projection_blocked_reads != expected_blocked_reads:
        raise AssertionError(
            "Gate/Up closed domains did not preserve the page/slice-group "
            f"boundaries: observed={ffn_projection_blocked_reads}, "
            f"expected={expected_blocked_reads}"
        )
    expected_hazard_waves = {
        10: {2: 16, 46: 16},
        11: {2: 16, 46: 16},
    }
    if ffn_projection_hazard_waves != expected_hazard_waves:
        raise AssertionError(
            "Gate/Up closed domains lost the real two-buffer reuse hazard: "
            f"observed={ffn_projection_hazard_waves}, "
            f"expected={expected_hazard_waves}"
        )
    required_swish_routes = {
        ("west", 6), ("west", 7), ("east", 14), ("east", 15),
    }
    total_swish_workload = sum(swish_output_workloads.values())
    invalid_swish_coverage = {}
    for output, workload in swish_output_workloads.items():
        observed = swish_write_workloads.get(output, {})
        mismatches = {
            route: observed.get(route, 0)
            for route in required_swish_routes
            if observed.get(route, 0) != workload
        }
        if mismatches:
            invalid_swish_coverage[output] = {
                "output_workload": workload,
                "write_workloads": mismatches,
            }
    if (args.ffn_schedule == "fused"
            and (total_swish_workload <= 0 or invalid_swish_coverage)):
        raise AssertionError(
            "FFN Swish queue-7 output workload is not fully consumed by "
            "W6/W7 and E14/E15 hidden writes: "
            f"total_output_workload={total_swish_workload}, "
            f"invalid={invalid_swish_coverage}"
        )
    if args.ffn_schedule == "fused":
        if ffn_input_allocation is None or not fused_hidden_locations:
            raise AssertionError(
                "cannot verify fused FFN input/hidden physical lifetimes"
            )
        _, input_bank, input_base, input_rows, input_slices = (
            ffn_input_allocation
        )
        input_end = input_base + input_rows
        for hidden_bank, hidden_base, hidden_slices in fused_hidden_locations:
            if (hidden_bank == input_bank
                    and not hidden_slices.isdisjoint(input_slices)
                    and input_base <= hidden_base < input_end):
                raise AssertionError(
                    "Fused Swish overwrites a live Gate activation: "
                    f"bank={hidden_bank}, row={hidden_base}, "
                    f"slices={sorted(hidden_slices & input_slices)}"
                )
    if up_first >= gate_first:
        raise AssertionError(
            "FFN did not schedule the resident Up projection before the "
            f"refilled Gate projection: up={up_first}, gate={gate_first}"
        )
    second_rms_end = rmsnorm_restore_ends[-1]
    if up_first - second_rms_end > 256:
        raise AssertionError(
            "FFN retained a large idle window after the second RMSNorm: "
            f"rms_end={second_rms_end}, first_up={up_first}"
        )
    if bias_bindings != {"query_bias", "key_bias", "value_bias"}:
        raise AssertionError(
            f"Schedule IR has incomplete projection bias bindings: {bias_bindings}"
        )
    expected_ready_cycles = {
        ("attention.value_cache" if args.kv_cache_capacity
         else "attention.value"): rope_interval[1],
        "attention.context": pv_interval[1],
    }
    if args.kv_cache_capacity:
        expected_ready_cycles["attention.key_cache"] = rope_interval[1]
        expected_states = {
            "attention.key_cache": ("state.kv.key", 65536),
            "attention.value_cache": ("state.kv.value", 65537),
        }
        if state_bindings != expected_states:
            raise AssertionError(
                "Schedule IR has incorrect KV state bindings: "
                f"observed={state_bindings}, expected={expected_states}"
            )
        if not {65536, 65537}.issubset(internal_address_bindings):
            raise AssertionError(
                "Schedule IR does not relocate both persistent KV states: "
                f"{sorted(internal_address_bindings)}"
            )
    for name, expected in expected_ready_cycles.items():
        observed = binding_ready_cycles.get(name)
        if observed != expected:
            raise AssertionError(
                f"{name} readiness is not on the global schedule timeline: "
                f"observed={observed}, expected={expected}"
            )
    if not qk_bias_low_fma or not qk_bias_high_fma:
        raise AssertionError(
            "Schedule IR does not fuse Q/K bias into both RoPE product chains"
        )
    if not value_bias_add:
        raise AssertionError(
            "Schedule IR does not route V projection through VXM bias add"
        )
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
    qkv_overlap_domains = [
        domain for domain in qkv_overlap_compute_domains
        if qkv_interval[0] <= int(domain["cycle"]) < qkv_interval[1]
    ]
    malformed_overlap_domains = [
        domain for domain in qkv_overlap_domains
        if (domain["repeat_count"], domain["repeat_interval"],
            domain["wave_count"], domain["wave_interval"],
            domain["group_count"], domain["group_interval"],
            domain["toggle"]) != (32, 1, 1, 32, 48, 32, True)
    ]
    if not qkv_overlap_domains or malformed_overlap_domains:
        raise AssertionError(
            "QKV/RoPE overlap must use one continuous 48-reduction MXM "
            "domain on activation streams 4/5 with two-buffer toggling: "
            f"domains={qkv_overlap_domains}, "
            f"malformed={malformed_overlap_domains}"
        )
    continuous_overlap_intervals: list[tuple[int, int]] = []
    for domain in qkv_overlap_domains:
        starts = [
            int(domain["cycle"]) + group * int(domain["group_interval"])
            for group in range(int(domain["group_count"]))
        ]
        duration = ((int(domain["repeat_count"]) - 1)
                    * int(domain["repeat_interval"]) + 1)
        intervals = [(start, start + duration) for start in starts]
        if any(end != next_start for (_, end), (next_start, _) in zip(
                intervals, intervals[1:])):
            raise AssertionError(
                "QKV overlap MXM reductions are not contiguous at 32-cycle "
                f"boundaries: unit={domain['unit_id']}, "
                f"intervals={intervals[:8]}"
            )
        continuous_overlap_intervals.extend(intervals)
    combine_overlaps = [
        (combine, compute)
        for combine in rope_combine_intervals
        for compute in continuous_overlap_intervals
        if max(combine[0], compute[0]) < min(combine[1], compute[1])
    ]
    if not combine_overlaps:
        raise AssertionError(
            "QKV continuous MXM domain does not cross a VXM RoPE combine "
            "window"
        )
    o_proj_accumulator_reads = [
        interval for interval in accumulator_read_intervals
        if (interval[0] < o_proj_interval[1]
            and interval[1] > o_proj_interval[0])
    ]
    if o_proj_accumulator_reads:
        raise AssertionError(
            "O projection retained standalone accumulator reads: "
            f"{o_proj_accumulator_reads[:4]}"
        )
    o_proj_streaming_computes = [
        interval for interval in streaming_bf16_compute_intervals
        if (interval[0] < o_proj_interval[1]
            and interval[1] > o_proj_interval[0])
    ]
    if not o_proj_streaming_computes:
        raise AssertionError(
            "O projection does not stream its final BF16 accumulations from "
            "the final MXM partial"
        )
    first_rope, first_mxm = overlaps[0]
    print(
        "QKV/bias-RoPE overlap: "
        f"{len({rope for rope, _ in overlaps})} RoPE product windows; "
        f"first RoPE {first_rope} with MXM {first_mxm}",
        flush=True,
    )
    print(
        "FFN resident-first: "
        f"second RMS end={second_rms_end}, first Up read={up_first}, "
        f"first Gate read={gate_first}",
        flush=True,
    )

    run([
        str(args.opt), "--input", str(schedule), "--output", str(command),
        "--pipeline", "ftlpu-schedule-to-commands", *common,
    ], "closed-form-schedule-to-hardware-commands")
    command_text = command.read_text(encoding="utf-8")
    expected_weight_pages = {
        2: 1,   # query projection
        3: 1,   # key projection
        4: 1,   # value projection
        5: 1,   # output projection
        10: 2,  # gate projection
        11: 2,  # up projection
        12: 12,  # down projection
    }
    binding_page_banks: dict[int, list[int]] = {}
    for line in command_text.splitlines():
        if ("ftlpu.command.binding" not in line
                or 'access = "input"' not in line
                or "paged_weight = true" not in line):
            continue
        binding = integer_attr(line, "index")
        count = integer_attr(line, "page_count")
        banks_match = re.search(r"page_banks = \[([^]]*)\]", line)
        if banks_match:
            banks = [int(value) for value in
                     re.findall(r"\d+", banks_match.group(1))]
        else:
            initial_bank = integer_attr(line, "bank")
            bank_count = integer_attr(line, "page_bank_count", 1)
            banks = [
                (initial_bank if bank_count == 1
                 else (initial_bank + page) % bank_count)
                for page in range(count)
            ]
        if len(banks) != count:
            raise AssertionError(
                "Qwen2.5 paged binding has incomplete physical bank "
                f"placement: binding={binding}, count={count}, banks={banks}"
            )
        binding_page_banks[binding] = banks
    observed_weight_pages: dict[int, dict[int, tuple[int, int, int]]] = {}
    for line in command_text.splitlines():
        if "ftlpu.command.weight_page" not in line:
            continue
        binding = integer_attr(line, "binding_index")
        page = integer_attr(line, "page_index")
        bank = integer_attr(line, "bank")
        ready = integer_attr(line, "ready_cycle")
        release = integer_attr(line, "release_cycle")
        pages = observed_weight_pages.setdefault(binding, {})
        if page in pages:
            raise AssertionError(
                "Qwen2.5 emitted a duplicate dynamic weight-page use: "
                f"binding={binding}, page={page}"
            )
        if ready >= release:
            raise AssertionError(
                "Qwen2.5 dynamic weight page has an empty residency window: "
                f"binding={binding}, page={page}, ready={ready}, "
                f"release={release}"
            )
        pages[page] = (bank, ready, release)
    observed_page_counts = {
        binding: len(pages)
        for binding, pages in observed_weight_pages.items()
    }
    if observed_page_counts != expected_weight_pages:
        raise AssertionError(
            "Qwen2.5 direct lowering lost its dynamic C2C weight pages: "
            f"observed={observed_page_counts}, expected={expected_weight_pages}"
        )
    if set(binding_page_banks) != set(expected_weight_pages):
        raise AssertionError(
            "Qwen2.5 dynamic page uses and paged bindings disagree: "
            f"bindings={sorted(binding_page_banks)}, "
            f"uses={sorted(observed_weight_pages)}"
        )
    for binding, expected_count in expected_weight_pages.items():
        expected_indices = set(range(expected_count))
        observed_indices = set(observed_weight_pages[binding])
        if observed_indices != expected_indices:
            raise AssertionError(
                "Qwen2.5 dynamic weight pages are not contiguous: "
                f"binding={binding}, pages={sorted(observed_indices)}"
            )
        for page, (bank, _, _) in observed_weight_pages[binding].items():
            expected_bank = binding_page_banks[binding][page]
            if bank != expected_bank:
                raise AssertionError(
                    "Qwen2.5 dynamic weight-page use does not match its "
                    "physical binding placement: "
                    f"binding={binding}, page={page}, bank={bank}, "
                    f"expected_bank={expected_bank}"
                )
    raw_counts = {
        name: command_text.count(f"ftlpu.command.{name}")
        for name in (
            "mem_3d", "mem_write_read_2d", "mxm_load_3d", "mxm_dequant_3d",
            "mxm_compute_3d", "vxm_run_2d", "sxm_run_2d",
        )
    }
    if any(count == 0 for count in raw_counts.values()):
        raise AssertionError(
            f"Qwen2.5 direct lowering is missing raw FU packets: {raw_counts}"
        )
    total_raw_domains = sum(raw_counts.values())
    # Aliased MEM read/write regions are emitted as separate closed domains so a
    # physical MEM ICU never has to interleave two active coarse instructions.
    if raw_counts["mem_3d"] > 19_600 or total_raw_domains > 21_250:
        raise AssertionError(
            "Qwen2.5 direct-domain lowering regressed its coarse-instruction "
            f"budget: total={total_raw_domains}, raw={raw_counts}"
        )
    for legacy in ("mem ", "mem_bundle ", "mxm ", "mxm_dequant ",
                   "vxm ", "sxm "):
        if f"ftlpu.command.{legacy}" in command_text:
            raise AssertionError(
                f"Qwen2.5 retained legacy fine command {legacy.strip()}"
            )

    run([
        str(args.compile), "--input", str(command),
        "--output", str(binary), "--input-stage", "command",
        "--target-config", str(args.target_config),
        "--mxm-execution", "vector",
        "--weight-bank", str(args.weight_bank),
        *(["--kv-cache-capacity", str(args.kv_cache_capacity)]
          if args.kv_cache_capacity else []),
    ], "hardware-commands-to-binary")
    binary_bytes = binary.stat().st_size
    if binary_bytes < 64:
        raise AssertionError("Qwen decoder-layer binary is unexpectedly small")
    if binary_bytes > 1_300_000:
        raise AssertionError(
            "Qwen2.5 direct-domain binary regressed its 1.30 MB budget: "
            f"{binary_bytes} bytes"
        )
    print(
        f"Qwen2.5-1.5B decoder-layer executable: {binary} "
        f"({binary_bytes} bytes, dynamic_weight_pages="
        f"{sum(expected_weight_pages.values())}, raw={raw_counts})", flush=True,
    )


if __name__ == "__main__":
    main()
