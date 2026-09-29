#!/usr/bin/env python3
"""Precompile exact, tile-aligned Qwen2.5 decode programs and write a manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ALIGNMENT = 32


def checked(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    build = repo / "build-ftlpu-vs2026-direct"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--past-lengths", type=int, nargs="+", default=(32, 64, 128, 224)
    )
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "results" / "qwen2_5_decoder_layer_decode_buckets",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "results" / "qwen2_5_decoder_layer_decode"
        / "decoder.stablehlo.mlir",
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=workspace / "FTLPU-CMODEL" / "config" / "ftlpu-lpu32.json",
    )
    parser.add_argument("--opt", type=Path, default=build / "compiler" / "ftlpu_opt.exe")
    parser.add_argument(
        "--compile", type=Path, default=build / "compiler" / "ftlpu-compile.exe"
    )
    parser.add_argument(
        "--runtime-test",
        type=Path,
        default=build / "runtime"
        / "compiled_qwen_real_decoder_layer_runtime_test.exe",
    )
    parser.add_argument(
        "--icu-export",
        type=Path,
        default=build / "runtime" / "ftlpu_icu_program_export.exe",
    )
    args = parser.parse_args()

    lengths = sorted(set(args.past_lengths))
    if not lengths:
        raise ValueError("at least one decode bucket is required")
    for past_len in lengths:
        if past_len <= 0 or past_len % ALIGNMENT:
            raise ValueError(
                f"past length {past_len} must be a positive multiple of {ALIGNMENT}"
            )
        if past_len + 1 > args.capacity:
            raise ValueError(
                f"past length {past_len} plus the current token exceeds "
                f"capacity {args.capacity}"
            )
    for path in (
        args.input,
        args.target_config,
        args.opt,
        args.compile,
        args.runtime_test,
        args.icu_export,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compile_test = repo / "compiler" / "tests" / "qwen2_5_1_5b_decode_cmodel_test.py"
    fixture_generator = (
        repo / "compiler" / "tests" / "qwen2_5_1_5b_decode_cmodel_fixture.py"
    )
    buckets: list[dict[str, object]] = []
    for past_len in lengths:
        bucket_dir = output / f"past_{past_len:04d}"
        fixture_dir = bucket_dir / "fixture"
        checked(
            [
                sys.executable,
                "-B",
                str(compile_test),
                "--opt",
                str(args.opt),
                "--compile",
                str(args.compile),
                "--runtime-test",
                str(args.runtime_test),
                "--icu-export",
                str(args.icu_export),
                "--input",
                str(args.input),
                "--target-config",
                str(args.target_config),
                "--output-dir",
                str(bucket_dir),
                "--past-len",
                str(past_len),
                "--current-len",
                "1",
                "--kv-cache-capacity",
                str(args.capacity),
                "--compile-only",
            ]
        )
        checked(
            [
                sys.executable,
                "-B",
                str(fixture_generator),
                "--output-dir",
                str(fixture_dir),
                "--past-len",
                str(past_len),
                "--capacity",
                str(args.capacity),
            ]
        )
        program = bucket_dir / "decoder.ftlpu"
        buckets.append(
            {
                "past_len": past_len,
                "current_len": 1,
                "position_offset": past_len,
                "resident_kv_tokens": past_len + 1,
                "program": relative(program, output),
                "fixture": relative(fixture_dir, output),
                "program_bytes": program.stat().st_size,
            }
        )

    manifest = {
        "schema_version": 1,
        "model": "Qwen2.5-1.5B decoder layer",
        "decode_layout": "native4",
        "selection_policy": "exact_past_length",
        "past_length_alignment": ALIGNMENT,
        "current_len": 1,
        "kv_cache_capacity": args.capacity,
        "note": (
            "Programs encode the RoPE position and attention domain. A request must "
            "match past_len exactly; selecting a larger covering bucket is invalid."
        ),
        "buckets": buckets,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
