#!/usr/bin/env python3
"""Build and run one Qwen2.5-1.5B decode token through ICU/CModel."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def run(
    command: list[str], phase: str, *, environment: dict[str, str] | None = None
) -> None:
    print(f"[{phase}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=environment)


def require(text: str, marker: str, artifact: Path) -> None:
    if marker not in text:
        raise AssertionError(f"{artifact} is missing {marker!r}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--icu-export", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "results" / "qwen2_5_decoder_layer_decode",
    )
    parser.add_argument("--weight-bank", type=int, choices=(0, 1), default=1)
    parser.add_argument("--past-len", type=int, default=32)
    parser.add_argument("--current-len", type=int, default=1)
    parser.add_argument("--kv-cache-capacity", type=int, default=256)
    parser.add_argument(
        "--projection-rope-overlap", choices=("on", "off"), default="off"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="stop after producing and structurally checking the ICU binary",
    )
    args = parser.parse_args()
    if args.past_len <= 0 or args.current_len <= 0:
        raise ValueError("decode lengths must be positive")
    if args.past_len % 32 != 0:
        raise ValueError("decode bucket past length must be a multiple of 32")
    if args.current_len != 1:
        raise ValueError("the decode bucket fixture currently emits one token")
    if args.past_len + args.current_len > args.kv_cache_capacity:
        raise ValueError("decode token does not fit in the KV-cache capacity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stablehlo = args.output_dir / "decoder.stablehlo.mlir"
    stream = args.output_dir / "decoder.stream.mlir"
    schedule = args.output_dir / "decoder.schedule.mlir"
    command = args.output_dir / "decoder.command.mlir"
    binary = args.output_dir / "decoder.ftlpu"
    linked_binary = args.output_dir / "decoder.linked.ftlpu"
    fixture = args.output_dir / "fixture"
    pre_execution = args.output_dir / "pre_execution"
    kv_dump = args.output_dir / "kv"
    runtime_icu_programs = args.output_dir / "runtime_icu_programs"
    runtime_pipeline = args.output_dir / "pipeline.csv"
    source = args.input.read_text(encoding="utf-8")
    # Decode has one logical token. Keep the seq32 source as the prefill
    # reference, but make the decode ABI and all current-token operations
    # genuinely 1xH rather than hiding 31 padded rows in the public tensor.
    source = source.replace("decoder_layer_seq32", "decoder_layer_decode1")
    source = source.replace("tensor<32x", "tensor<1x")
    source = source.replace("x32x", "x1x")
    source = source.replace("tensor<12x1x32", "tensor<12x1x1")
    source = source.replace("array<i64: 32,", "array<i64: 1,")
    stablehlo.write_text(source, encoding="utf-8")

    common = [
        "--mxm-execution", "native4",
        "--ffn-schedule", "fused",
        "--projection-rope-overlap", args.projection_rope_overlap,
        "--target-config", str(args.target_config),
        "--weight-bank", str(args.weight_bank),
        "--kv-cache-capacity", str(args.kv_cache_capacity),
        "--decode-past-len", str(args.past_len),
        "--decode-current-len", str(args.current_len),
    ]
    run([
        str(args.opt), "--input", str(stablehlo), "--output", str(stream),
        "--pipeline", "ftlpu-stablehlo-to-stream", *common,
    ], "stablehlo-to-decode-stream")
    run([
        str(args.opt), "--input", str(stream), "--output", str(schedule),
        "--pipeline", "ftlpu-stream-to-schedule", *common,
    ], "decode-stream-to-schedule")
    run([
        str(args.opt), "--input", str(schedule), "--output", str(command),
        "--pipeline", "ftlpu-schedule-to-commands", *common,
    ], "decode-schedule-to-icu-commands")
    run([
        str(args.compile), "--input", str(command),
        "--output", str(binary), "--input-stage", "command",
        "--target-config", str(args.target_config),
        "--mxm-execution", "native4",
        "--weight-bank", str(args.weight_bank),
        "--kv-cache-capacity", str(args.kv_cache_capacity),
    ], "icu-commands-to-binary")

    stream_text = stream.read_text(encoding="utf-8")
    require(stream_text, f"ftlpu.decode_past_len = {args.past_len} : i64", stream)
    require(stream_text,
            f"ftlpu.decode_current_len = {args.current_len} : i64", stream)
    require(stream_text, 'ftlpu.mxm_execution_policy = "native4"', stream)
    require(stream_text, "tensor<1x1536xbf16>", stream)
    require(stream_text, 'kind = "fp16_rope_table_decode_compact"', stream)
    rope_mirror = re.search(
        r"rope_mirror = \{[^}]*base_row = (\d+) : i64", stream_text
    )
    physical_input_rows = 4 * (1536 // 32)
    if rope_mirror is None or int(rope_mirror.group(1)) < physical_input_rows:
        raise AssertionError(
            f"{stream} places the initialized RoPE mirror inside the live "
            f"one-token activation tile [0, {physical_input_rows})"
        )
    schedule_text = schedule.read_text(encoding="utf-8")
    if not re.search(
        r'ftlpu\.schedule\.binding %arg0 .*instruction_count = 192 : i64, '
        r'kind = "fp16_mxm_distributed_16"',
        schedule_text,
    ):
        raise AssertionError(
            f"{schedule} does not reserve the full 16-slice physical input span"
        )
    compute_domains = [
        line for line in schedule_text.splitlines()
        if "ftlpu.schedule.mxm_issue" in line and 'opcode = "compute"' in line
    ]
    if not any(
        "group_count = 24 : i64" in line
        and "wave_count = 48 : i64" in line
        and "wave_interval = 6 : i64" in line
        for line in compute_domains
    ):
        raise AssertionError(
            f"{schedule} does not use the compact 6-cycle Q projection cadence"
        )
    if not any(
        "group_count = 24 : i64" in line
        and "wave_count = 48 : i64" in line
        and "wave_interval = 8 : i64" in line
        for line in compute_domains
    ):
        raise AssertionError(
            f"{schedule} does not use the legal 8-cycle O projection cadence"
        )
    schedule_lines = schedule_text.splitlines()
    key_start = next(
        index for index, line in enumerate(schedule_lines)
        if "ftlpu.schedule.mem_transfer" in line
        and "address_binding = 3 : i64" in line
        and 'opcode = "read"' in line
    )
    output_start = next(
        index for index, line in enumerate(schedule_lines[key_start:], key_start)
        if "ftlpu.schedule.mem_transfer" in line
        and "address_binding = 5 : i64" in line
        and 'opcode = "read"' in line
    )
    key_section = schedule_lines[key_start:output_start]

    def integer_attr(line: str, name: str, default: int = -1) -> int:
        match = re.search(rf"\b{re.escape(name)} = (-?\d+) : i64", line)
        return int(match.group(1)) if match else default

    native4_compute_cycles: dict[int, list[int]] = {}
    for line in schedule_lines:
        if (
            "ftlpu.schedule.mxm_issue" not in line
            or 'decode_layout = "native4"' not in line
            or 'opcode = "decode_stream_compute"' not in line
            or integer_attr(line, "unit_id") != 0
        ):
            continue
        repeat_count = integer_attr(line, "repeat_count")
        native4_compute_cycles.setdefault(repeat_count, []).append(
            integer_attr(line, "cycle")
        )
    for local_waves in (140, 24):
        cycles = native4_compute_cycles.get(local_waves, [])
        adjacent_deltas = [
            later - earlier for earlier, later in zip(cycles, cycles[1:])
            if 0 < later - earlier < 1000
        ]
        expected_interval = local_waves + 4
        if not adjacent_deltas or min(adjacent_deltas) != expected_interval:
            raise AssertionError(
                "native4 MXM reductions retain an avoidable pipeline bubble: "
                f"waves={local_waves}, deltas={sorted(set(adjacent_deltas))[:8]}, "
                f"expected={expected_interval}"
            )

    key_raw_queues = {
        (integer_attr(line, "hemisphere"), integer_attr(line, "slice"),
         integer_attr(line, "bank"))
        for line in key_section
        if "ftlpu.schedule.mem_transfer" in line
        and 'opcode = "write"' in line
        and integer_attr(line, "packed_stream") in (32, 33)
        and integer_attr(line, "bank") == 1
        and integer_attr(line, "slice") < 8
    }
    expected_key_raw_queues = {
        *((0, slice_id, 1) for slice_id in range(4)),
        *((1, slice_id, 1) for slice_id in range(4, 8)),
    }
    if key_raw_queues != expected_key_raw_queues:
        raise AssertionError(
            "decode Key projection omitted a raw staging half: "
            f"observed={sorted(key_raw_queues)}, "
            f"expected={sorted(expected_key_raw_queues)}"
        )
    output_weight_reads = [
        line for line in schedule_lines
        if "ftlpu.schedule.mem_transfer" in line
        and "address_binding = 5 : i64" in line
        and 'opcode = "read"' in line
    ]
    output_weight_queues = {
        (integer_attr(line, "hemisphere"), integer_attr(line, "slice"))
        for line in output_weight_reads
    }
    if len(output_weight_reads) != 16 or len(output_weight_queues) != 16 or not all(
        "group_count = 24 : i64" in line
        and "wave_count = 48 : i64" in line
        and "wave_interval = 8 : i64" in line
        for line in output_weight_reads
    ):
        raise AssertionError(
            f"{schedule} does not keep one closed-form O-weight READ_3D per "
            "hemisphere/ICU"
        )
    command_text = command.read_text(encoding="utf-8")
    require(command_text, 'ftlpu.command_lowering = "direct"', command)
    require(command_text, 'role = "state.kv.key"', command)
    require(command_text, 'role = "state.kv.value"', command)
    require(command_text, "ftlpu.command.mem_3d", command)
    require(command_text, 'opcode = "decode_load_activation"', command)
    require(command_text, 'opcode = "decode_stream_compute"', command)
    require(command_text, 'decode_layout = "native4"', command)
    require(command_text, "ftlpu.command.vxm_run_2d", command)
    require(command_text, "ftlpu.command.weight_page", command)

    ffn_bank_b = (args.weight_bank + 1) % 2
    expected_ffn_banks = {
        10: ffn_bank_b,       # Gate: opposite the Attention epoch.
        11: args.weight_bank, # Up: alternate bank while Gate computes.
        12: ffn_bank_b,       # Down: alternate back while Up computes.
    }
    for binding_index, expected_bank in expected_ffn_banks.items():
        line = next(
            line for line in command_text.splitlines()
            if "ftlpu.command.binding" in line
            and f"index = {binding_index} : i64" in line
            and "paged_weight = true" in line
        )
        banks_match = re.search(r"\bpage_banks = \[([^]]*)\]", line)
        if banks_match is None:
            raise AssertionError(
                f"decode FFN binding {binding_index} has no physical page bank"
            )
        banks = [
            int(value.strip())
            for value in banks_match.group(1).split(",")
            if value.strip()
        ]
        if banks != [expected_bank] or integer_attr(line, "bank") != expected_bank:
            raise AssertionError(
                "decode FFN epochs do not ping-pong physical SRAM banks: "
                f"binding={binding_index}, placement_bank="
                f"{integer_attr(line, 'bank')}, page_banks={banks}, "
                f"expected={[expected_bank]}"
            )
        if binding_index == 10 and "runtime_prefetch = true" not in line:
            raise AssertionError(
                "decode Gate page was folded into the Attention startup "
                "barrier instead of being prefetched while Attention runs"
            )

    paged_weight_lines = [
        line for line in command_text.splitlines()
        if "ftlpu.command.binding" in line and "paged_weight = true" in line
    ]
    for line in paged_weight_lines:
        storage_match = re.search(r"\bpage_storage_slices = \[([^]]*)\]", line)
        if storage_match is None:
            raise AssertionError("paged decode weight has no physical slice pool")
        storage_slices = {
            int(value.strip())
            for value in storage_match.group(1).split(",")
            if value.strip()
        }
        if not storage_slices or min(storage_slices) < 20:
            raise AssertionError(
                "decode weight page escaped the dedicated weight MEM slices: "
                f"{sorted(storage_slices)}"
            )

    probability_pack_line = next(
        line for line in command_text.splitlines()
        if "ftlpu.command.binding" in line
        and 'name = "attention.probability_pack"' in line
    )
    probability_pack_slices_match = re.search(
        r"\bslices = \[([^]]*)\]", probability_pack_line
    )
    if probability_pack_slices_match is None:
        raise AssertionError("decode probability pack has no physical slices")
    probability_pack_slices = {
        int(value.strip())
        for value in probability_pack_slices_match.group(1).split(",")
        if value.strip()
    }
    if (
        integer_attr(probability_pack_line, "bank") != args.weight_bank
        or not probability_pack_slices
        or min(probability_pack_slices) < 20
    ):
        raise AssertionError(
            "decode probability pack must borrow the released Attention "
            "weight bank, not overwrite the prefetched Gate bank: "
            f"bank={integer_attr(probability_pack_line, 'bank')}, "
            f"slices={sorted(probability_pack_slices)}"
        )

    def binding_placement(name: str) -> tuple[int, int, int, set[int]]:
        line = next(
            line for line in command_text.splitlines()
            if "ftlpu.command.binding" in line and f'name = "{name}"' in line
        )
        bank = integer_attr(line, "bank")
        base = integer_attr(line, "base_row")
        count = integer_attr(line, "instruction_count")
        slices_match = re.search(r"\bslices = \[([^]]*)\]", line)
        if bank < 0 or base < 0 or count <= 0 or slices_match is None:
            raise AssertionError(f"cannot parse {name} placement from {command}")
        slices = {
            int(value.strip())
            for value in slices_match.group(1).split(",")
            if value.strip()
        }
        return bank, base, base + count, slices

    rope_bank, rope_begin, rope_end, rope_slices = binding_placement(
        "rope.cos_sin"
    )
    key_bank, key_begin, key_end, key_slices = binding_placement(
        "attention.key_cache"
    )
    if (
        rope_bank == key_bank
        and not rope_slices.isdisjoint(key_slices)
        and max(rope_begin, key_begin) < min(rope_end, key_end)
    ):
        raise AssertionError(
            "decode RoPE table overlaps persistent Key cache: "
            f"rope=bank{rope_bank}{sorted(rope_slices)}[{rope_begin},{rope_end}), "
            f"key=bank{key_bank}{sorted(key_slices)}[{key_begin},{key_end})"
        )

    if args.compile_only:
        print(
            "Qwen2.5-1.5B decode bucket compiled: "
            f"past={args.past_len}, current={args.current_len}, "
            f"capacity={args.kv_cache_capacity}, binary={binary}",
            flush=True,
        )
        return

    fixture_script = Path(__file__).with_name(
        "qwen2_5_1_5b_decode_cmodel_fixture.py"
    )
    run([
        sys.executable, str(fixture_script),
        "--output-dir", str(fixture),
        "--capacity", str(args.kv_cache_capacity),
        "--past-len", str(args.past_len),
    ], "generate-ddr-fixture")
    runtime_environment = os.environ.copy()
    runtime_environment["FTLPU_QWEN_C2C_LINKED_BINARY"] = str(
        linked_binary.resolve()
    )
    runtime_environment["FTLPU_QWEN_C2C_PRE_EXECUTION_DIR"] = str(
        pre_execution.resolve()
    )
    runtime_environment["FTLPU_QWEN_KV_DUMP_DIR"] = str(kv_dump.resolve())
    runtime_environment["FTLPU_QWEN_PIPELINE_CSV"] = str(
        runtime_pipeline.resolve()
    )
    run(
        [str(args.runtime_test), str(binary), str(fixture)],
        "icu-cmodel-decode",
        environment=runtime_environment,
    )
    with runtime_pipeline.open(newline="", encoding="utf-8") as stream:
        pipeline_rows = list(csv.DictReader(stream))
    attention_prefetch_end = max(
        int(row["end"])
        for row in pipeline_rows
        if row["resource"] == "C2C.E.Prefetch"
        and "phase=pre_execution" in row["detail"]
    )
    gate_prefetch = next(
        row for row in pipeline_rows
        if row["resource"] == "C2C.E.Prefetch"
        and "bindings=10 " in row["detail"]
    )
    first_mxm_issue = min(
        int(row["start"])
        for row in pipeline_rows
        if row["resource"] == "MXM.E0.Load"
    )
    first_gate_wait = min(
        int(row["start"])
        for row in pipeline_rows
        if row["resource"] == "ICU.PageReadyWait"
    )
    gate_prefetch_start = int(gate_prefetch["start"])
    gate_prefetch_end = int(gate_prefetch["end"])
    if not (
        attention_prefetch_end <= first_mxm_issue < gate_prefetch_end
        and gate_prefetch_start < first_gate_wait
    ):
        raise AssertionError(
            "Attention execution waited for the Gate page instead of "
            "overlapping Gate prefetch: "
            f"attention_ready={attention_prefetch_end}, "
            f"first_mxm={first_mxm_issue}, gate_prefetch="
            f"[{gate_prefetch_start}, {gate_prefetch_end}), "
            f"gate_wait={first_gate_wait}"
        )
    run([
        str(args.icu_export), str(linked_binary), str(runtime_icu_programs),
        "--pre-execution-dir", str(pre_execution),
    ], "export-runtime-icu-programs")

    print(
        "Qwen2.5-1.5B ICU/CModel decode passed: "
        f"past={args.past_len}, current={args.current_len}, "
        f"capacity={args.kv_cache_capacity}, binary={binary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
