#!/usr/bin/env python3
"""Imports and packages a reusable-executable Hugging Face decoder stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_phase(name: str, command: list[str]) -> None:
    start = time.perf_counter()
    print(f"{name}...", flush=True)
    subprocess.run(command, check=True)
    print(f"{name}: {time.perf_counter() - start:.2f}s", flush=True)


def compile_executables(
    *,
    opt: Path,
    translate: Path,
    stablehlo: Path,
    target_configs: list[Path],
    output_dir: Path,
    ffn_schedule: str,
    rmsnorm_strategy: str,
    mxm_execution: str,
    reuse_executables: bool,
    weight_banks: list[int | None],
) -> list[Path]:
    executables: list[Path] = []
    for index, target_config in enumerate(target_configs):
        executable_dir = output_dir / "executables" / f"variant{index}"
        executable_dir.mkdir(parents=True, exist_ok=True)
        schedule_ir = executable_dir / "decoder_layer.schedule.mlir"
        command_ir = executable_dir / "decoder_layer.command.mlir"
        binary = executable_dir / "decoder_layer.ftlpu"
        if reuse_executables and binary.is_file():
            print(
                f"reuse executable variant {index}: {binary}",
                flush=True,
            )
            executables.append(binary)
            continue
        weight_bank_args = (
            ["--weight-bank", str(weight_banks[index])]
            if weight_banks[index] is not None else []
        )
        run_phase(
            f"lower executable variant {index} to Schedule IR",
            [
                str(opt),
                "--input", str(stablehlo),
                "--output", str(schedule_ir),
                "--pipeline", "ftlpu-stablehlo-to-schedule",
                "--ffn-schedule", ffn_schedule,
                "--rmsnorm-strategy", rmsnorm_strategy,
                "--mxm-execution", mxm_execution,
                "--target-config", str(target_config),
            ] + weight_bank_args,
        )
        run_phase(
            f"lower executable variant {index} to Command IR",
            [
                str(opt),
                "--input", str(schedule_ir),
                "--output", str(command_ir),
                "--pipeline", "ftlpu-schedule-to-commands",
                "--target-config", str(target_config),
            ],
        )
        run_phase(
            f"translate executable variant {index}",
            [
                str(translate),
                "--input", str(command_ir),
                "--output", str(binary),
            ],
        )
        executables.append(binary)
    return executables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--executable", type=Path, action="append",
        help="precompiled reusable decoder executable",
    )
    parser.add_argument(
        "--opt", type=Path,
        help="ftlpu_opt used to compile decoder executable variants",
    )
    parser.add_argument(
        "--translate", type=Path,
        help="ftlpu-translate used to serialize executable variants",
    )
    parser.add_argument(
        "--stablehlo", type=Path,
        help="parameterized decoder-layer StableHLO input",
    )
    parser.add_argument(
        "--target-config", type=Path, action="append",
        help="target configuration for each resident-weight variant",
    )
    parser.add_argument(
        "--ffn-schedule", choices=("tail", "fused"), default="tail",
        help="FFN SwiGLU scheduling strategy used for compiled executables",
    )
    parser.add_argument(
        "--rmsnorm-strategy",
        choices=("vxm-square-mxm-reduce", "vxm-feedback"),
        default="vxm-feedback",
    )
    parser.add_argument(
        "--mxm-execution",
        choices=("auto", "vector", "legacy"),
        default="auto",
        help="MXM projection execution policy for every decoder variant",
    )
    parser.add_argument(
        "--layers-per-executable", type=int, default=6,
        help="contiguous layers assigned to each resident-weight variant",
    )
    parser.add_argument(
        "--c2c-weight-paging", action="store_true",
        help=(
            "compile alternating bank0/bank1 decoder executables and pack "
            "each layer as a C2C weight page"
        ),
    )
    parser.add_argument(
        "--pack-model-weights", type=Path,
        help="ftlpu-pack-model-weights used by --c2c-weight-paging",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument(
        "--reuse-golden", action="store_true",
        help="reuse already imported layer golden files in output-dir",
    )
    parser.add_argument(
        "--reuse-executables", action="store_true",
        help="reuse already compiled variant binaries in output-dir",
    )
    parser.add_argument(
        "--checkpoint-outputs",
        action="store_true",
        help="embed and download every decoder layer golden output",
    )
    parser.add_argument(
        "--ignore-attention-bias",
        action="store_true",
        help="omit checkpoint Q/K/V biases from import and golden generation",
    )
    parser.add_argument(
        "--final-rmsnorm-executable",
        type=Path,
        help="include host embedding, LPU final RMSNorm, and host LM head",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        help="number of decoder layers; defaults to the remaining HF layers",
    )
    args = parser.parse_args()

    config = json.loads(
        (args.model_dir / "config.json").read_text(encoding="utf-8")
    )
    total_layers = int(config["num_hidden_layers"])
    layer_count = (
        args.layer_count
        if args.layer_count is not None
        else total_layers - args.first_layer
    )
    if args.first_layer < 0 or layer_count <= 0:
        raise ValueError("decoder layer range must be non-empty")
    if args.first_layer + layer_count > total_layers:
        raise ValueError("decoder layer range exceeds the HF model")
    if args.layers_per_executable <= 0:
        raise ValueError("layers per executable must be positive")
    required_executables = min(2, layer_count) if args.c2c_weight_paging else (
        layer_count + args.layers_per_executable - 1
    ) // args.layers_per_executable
    if args.c2c_weight_paging and args.pack_model_weights is None:
        raise ValueError(
            "--pack-model-weights is required by --c2c-weight-paging"
        )
    compile_options = (
        args.opt, args.translate, args.stablehlo, args.target_config
    )
    compile_requested = any(value for value in compile_options)
    if compile_requested:
        if args.executable:
            raise ValueError(
                "--executable cannot be combined with compiler options"
            )
        if not all((args.opt, args.translate, args.stablehlo)):
            raise ValueError(
                "--opt, --translate, and --stablehlo are required when "
                "compiling executables"
            )
        if not args.target_config or len(args.target_config) not in (
            1, required_executables
        ):
            raise ValueError(
                "provide one target config or exactly "
                f"{required_executables} resident target configs"
            )
        target_configs = (
            args.target_config * required_executables
            if len(args.target_config) == 1
            else args.target_config
        )
        executables = compile_executables(
            opt=args.opt,
            translate=args.translate,
            stablehlo=args.stablehlo,
            target_configs=target_configs,
            output_dir=args.output_dir,
            ffn_schedule=args.ffn_schedule,
            rmsnorm_strategy=args.rmsnorm_strategy,
            mxm_execution=args.mxm_execution,
            reuse_executables=args.reuse_executables,
            weight_banks=(
                list(range(required_executables)) if args.c2c_weight_paging
                else [None] * required_executables
            ),
        )
    else:
        executables = args.executable or []
        if len(executables) not in (1, required_executables):
            raise ValueError(
                "provide one reusable executable or exactly "
                f"{required_executables} resident-weight variants"
            )

    tools_dir = Path(__file__).resolve().parent
    importer = tools_dir / "import_hf_decoder_layer.py"
    packager = tools_dir / "build_decoder_layer_package.py"
    golden_dirs: list[Path] = []
    preceding_output: Path | None = None

    for layer in range(args.first_layer, args.first_layer + layer_count):
        layer_dir = args.output_dir / f"layer{layer}_seq{args.seq_len}"
        required_files = (
            layer_dir / "metadata.json",
            layer_dir / "input.bf16.bin",
            layer_dir / "golden.bf16.bin",
        )
        if args.reuse_golden and all(path.is_file() for path in required_files):
            golden_dirs.append(layer_dir)
            preceding_output = layer_dir / "golden.bf16.bin"
            continue
        command = [
            sys.executable,
            str(importer),
            "--model-dir",
            str(args.model_dir),
            "--output-dir",
            str(layer_dir),
            "--layer",
            str(layer),
            "--seq-len",
            str(args.seq_len),
        ]
        if preceding_output is not None:
            command.extend(["--input-bf16", str(preceding_output)])
        if args.ignore_attention_bias:
            command.append("--ignore-attention-bias")
        run_phase(f"import HF layer {layer}", command)
        golden_dirs.append(layer_dir)
        preceding_output = layer_dir / "golden.bf16.bin"

    package_command = [sys.executable, str(packager)]
    for golden_dir in golden_dirs:
        package_command.extend(["--golden-dir", str(golden_dir)])
    if len(executables) == 1:
        layer_executables = executables
    elif args.c2c_weight_paging:
        layer_executables = [
            executables[layer_offset % 2]
            for layer_offset in range(layer_count)
        ]
    else:
        layer_executables = [
            executables[layer_offset // args.layers_per_executable]
            for layer_offset in range(layer_count)
        ]
    for executable in layer_executables:
        package_command.extend(["--executable", str(executable)])
    logical_output = (
        args.output_dir / "model.logical.ftlpum"
        if args.c2c_weight_paging else args.output
    )
    package_command.extend(["--output", str(logical_output)])
    if args.checkpoint_outputs:
        package_command.append("--checkpoint-outputs")
    if args.final_rmsnorm_executable:
        sys.path.insert(0, str(tools_dir))
        from import_hf_decoder_layer import SafeTensorStore, write_bf16

        store = SafeTensorStore(args.model_dir)
        boundary_dir = args.output_dir / "model_boundaries"
        boundary_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = boundary_dir / "embed_tokens.bf16.bin"
        final_norm_path = boundary_dir / "final_norm.bf16.bin"
        write_bf16(
            embedding_path, store.read("model.embed_tokens.weight")
        )
        write_bf16(final_norm_path, store.read("model.norm.weight"))
        package_command.extend([
            "--embedding-table-bf16", str(embedding_path),
            "--vocab-size", str(int(config["vocab_size"])),
            "--final-norm-bf16", str(final_norm_path),
            "--final-rmsnorm-executable",
            str(args.final_rmsnorm_executable),
        ])
    run_phase("build model package", package_command)
    if args.c2c_weight_paging:
        run_phase(
            "pack alternating C2C weight pages",
            [
                str(args.pack_model_weights),
                "--input", str(logical_output),
                "--output", str(args.output),
                "--first-bank", "0",
            ],
        )
        logical_output.unlink()


if __name__ == "__main__":
    main()
