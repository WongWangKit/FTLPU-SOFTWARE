#!/usr/bin/env python3
"""Build and run one Qwen2.5-1.5B decode token through ICU/CModel."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    if args.past_len <= 0 or args.current_len <= 0:
        raise ValueError("decode lengths must be positive")
    if args.past_len != 32 or args.current_len != 1:
        raise ValueError("the deterministic fixture currently models decode 32+1")
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
    output_weight_reads = [
        line for line in schedule_text.splitlines()
        if "ftlpu.schedule.mem_transfer" in line
        and "address_binding = 5 : i64" in line
        and 'opcode = "read"' in line
        and "slice = 20 : i64" in line
    ]
    if len(output_weight_reads) != 2 or not all(
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

    fixture_script = Path(__file__).with_name(
        "qwen2_5_1_5b_decode_cmodel_fixture.py"
    )
    run([
        sys.executable, str(fixture_script),
        "--output-dir", str(fixture),
        "--capacity", str(args.kv_cache_capacity),
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
