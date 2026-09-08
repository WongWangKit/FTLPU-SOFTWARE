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
    parser.add_argument("--ffn-schedule", choices=("tail", "fused"),
                        default="fused")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kv-cache-capacity", type=int, default=0)
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
        "--mxm-execution", "vector", "--ffn-schedule", args.ffn_schedule,
        "--target-config", str(args.target_config),
        "--weight-bank", str(args.weight_bank),
        "--rmsnorm-strategy", "vxm-feedback",
        "--icu-macro-schedule",
    ]
    if args.kv_cache_capacity:
        common += ["--kv-cache-capacity", str(args.kv_cache_capacity)]
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
    rope_interval: tuple[int, int] | None = None
    pv_interval: tuple[int, int] | None = None
    o_proj_interval: tuple[int, int] | None = None
    rmsnorm_restore_ends: list[int] = []
    ffn_weight_first_cycles: dict[int, int | None] = {10: None, 11: None}
    fused_swish_outputs: set[str] = set()
    fused_swish_writes: dict[str, set[tuple[str, int]]] = {}
    fused_hidden_locations: list[tuple[int, int, frozenset[int]]] = []
    ffn_input_allocation: tuple[str, int, int, int, frozenset[int]] | None = None
    rope_product_intervals: list[tuple[int, int]] = []
    qk_bias_low_fma = False
    qk_bias_high_fma = False
    value_bias_add = False
    bias_bindings: set[str] = set()
    binding_ready_cycles: dict[str, int] = {}
    state_bindings: dict[str, tuple[str, int]] = {}
    internal_address_bindings: set[int] = set()
    mxm_compute_intervals: list[tuple[int, int]] = []
    streaming_bf16_compute_intervals: list[tuple[int, int]] = []
    accumulator_read_intervals: list[tuple[int, int]] = []
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
                    'repeat_count = 32 : i64',
                )):
                    result = re.match(r"\s*(%\d+)\s*=", line)
                    if result:
                        fused_swish_outputs.add(result.group(1))
            if "ftlpu.schedule.mem_write" in line:
                source = re.search(r"ftlpu\.schedule\.mem_write\s+(%\d+)", line)
                if source and source.group(1) in fused_swish_outputs:
                    hemisphere = re.search(
                        r'placement = \{.*?hemisphere = "(east|west)"', line
                    )
                    if hemisphere:
                        fused_swish_writes.setdefault(source.group(1), set()).add(
                            (hemisphere.group(1), integer_attr(line, "stream_base"))
                        )
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
    if rope_interval is None:
        raise AssertionError("Schedule IR is missing the RoPE timeline")
    if pv_interval is None:
        raise AssertionError("Schedule IR is missing the PV timeline")
    if o_proj_interval is None:
        raise AssertionError("Schedule IR is missing the O projection timeline")
    if len(rmsnorm_restore_ends) < 2:
        raise AssertionError("Schedule IR is missing the second RMSNorm restore")
    gate_first = ffn_weight_first_cycles[10]
    up_first = ffn_weight_first_cycles[11]
    if gate_first is None or up_first is None:
        raise AssertionError("Schedule IR is missing Gate/Up paged weight reads")
    mirrored_swish_outputs = [
        output for output in fused_swish_outputs
        if {("west", 6), ("east", 14)}.issubset(
            fused_swish_writes.get(output, set())
        )
    ]
    if args.ffn_schedule == "fused" and not mirrored_swish_outputs:
        raise AssertionError(
            "Fused Swish does not consume both fixed VXM chain outputs "
            "through W6/W7 and E14/E15 hidden writes"
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
        str(args.compile), "--input", str(schedule),
        "--output", str(binary), "--input-stage", "schedule",
        "--target-config", str(args.target_config),
        "--mxm-execution", "vector",
        "--weight-bank", str(args.weight_bank),
        "--icu-macro-schedule",
        *(["--kv-cache-capacity", str(args.kv_cache_capacity)]
          if args.kv_cache_capacity else []),
    ], "compressed-schedule-to-binary")
    if binary.stat().st_size < 64:
        raise AssertionError("Qwen decoder-layer binary is unexpectedly small")
    print(
        f"Qwen2.5-1.5B decoder-layer executable: {binary} "
        f"({binary.stat().st_size} bytes)", flush=True,
    )


if __name__ == "__main__":
    main()
